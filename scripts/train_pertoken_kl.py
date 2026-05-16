"""Per-token KL distillation training (canonical Thinking Machines /
DeepSeek-V4 algorithm, top-K approximation).

Loss per example, per response position p:
    teacher_topk = {tok_id: logprob} for K teacher-top-K tokens at position p
    Renormalize teacher logprobs over the K tokens -> p_T (sums to 1)
    Gather student logits at the same K tokens, log-softmax -> log_p_S (sums in log to 0)
    KL(p_T || p_S) = sum_k p_T[k] * (log p_T[k] - log_p_S[k])

Loss is mean over response positions, over batch.

Uses QLoRA (4-bit base) on Qwen3-8B for memory efficiency. The same shape as
the SFT trainer (scripts/train_qwen_sft_onpolicy.py) — bf16 training, AdamW,
cosine schedule, ~3 epochs.

Environment variables:
  PL_MODEL_ID   base model path or HF id (default: Qwen/Qwen3-8B)
  PL_KL_PATH    teacher-topk jsonl produced by pertoken_kl_collect.py
  PL_OUT_DIR    adapter output dir
  PL_KL_EPOCHS  number of epochs (default 3)
  PL_KL_LR      learning rate (default 5e-5)
  PL_KL_BS      batch size, in examples (default 1; per-token KL is memory-heavy)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_ID = os.environ.get("PL_MODEL_ID", "Qwen/Qwen3-8B")
KL_PATH = os.environ["PL_KL_PATH"]
OUT_DIR = os.environ["PL_OUT_DIR"]
EPOCHS = int(os.environ.get("PL_KL_EPOCHS", "3"))
LR = float(os.environ.get("PL_KL_LR", "5e-5"))
BS = int(os.environ.get("PL_KL_BS", "1"))
GRAD_ACCUM = int(os.environ.get("PL_KL_GRAD_ACCUM", "8"))
MAX_LEN = int(os.environ.get("PL_KL_MAX_LEN", "4096"))
WARMUP = float(os.environ.get("PL_KL_WARMUP", "0.05"))
KL_TEMP = float(os.environ.get("PL_KL_TEMP", "1.0"))


def load_records(path: str, max_len: int) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if len(d["input_ids"]) > max_len:
                continue
            if not d.get("teacher_topk_logprobs"):
                continue
            if d.get("response_len", 0) < 2:
                continue
            out.append(d)
    return out


class PerTokenKLDataset(Dataset):
    def __init__(self, records: list[dict], max_len: int):
        self.records = records
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        input_ids = torch.tensor(r["input_ids"], dtype=torch.long)
        n = input_ids.shape[0]
        response_start = r["response_start"]
        topk_list = r["teacher_topk_logprobs"]
        K = max((len(d) for d in topk_list if d), default=20)
        # Pad each position's top-K to K with -inf
        teacher_ids = torch.full((len(topk_list), K), -1, dtype=torch.long)
        teacher_lps = torch.full((len(topk_list), K), float("-inf"), dtype=torch.float32)
        for i, d in enumerate(topk_list):
            if not d:
                continue
            entries = list(d.items())[:K]
            for j, (tid, lp) in enumerate(entries):
                teacher_ids[i, j] = int(tid)
                teacher_lps[i, j] = float(lp)
        return {
            "input_ids": input_ids,
            "response_start": response_start,
            "teacher_ids": teacher_ids,
            "teacher_lps": teacher_lps,
        }


def collate(batch):
    return batch  # we'll process examples one at a time inside step


def per_token_kl_loss(student_logits: torch.Tensor,
                        response_start: int,
                        teacher_ids: torch.Tensor,
                        teacher_lps: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compute mean per-token forward KL over response positions.

    student_logits: [seq_len, vocab]  (full sequence)
    teacher_ids:    [n_resp, K]
    teacher_lps:    [n_resp, K]       (raw teacher logprobs over top-K)

    For each response position i (0..n_resp-1):
      teacher_dist = softmax(teacher_lps[i, :]) over valid (non -inf) entries
      student_dist_on_topk = log_softmax(student_logits_at_pos[teacher_ids[i, :]])
      KL(teacher || student) = sum_k teacher_dist[k] * (log teacher_dist[k] - student_dist_on_topk[k])
    """
    n_resp = teacher_ids.shape[0]
    if n_resp == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    # student logits at positions [response_start-1 .. response_start+n_resp-2] predict
    # the response tokens. We need logits at those positions, of shape [n_resp, vocab].
    if response_start == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0
    start = response_start - 1
    end = start + n_resp
    end = min(end, student_logits.shape[0])
    actual_n = end - start
    if actual_n <= 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0
    s_logits = student_logits[start:end, :]  # [actual_n, vocab]

    t_ids = teacher_ids[:actual_n].to(student_logits.device)  # [actual_n, K]
    t_lps = teacher_lps[:actual_n].to(student_logits.device)  # [actual_n, K]
    valid_mask = (t_ids >= 0) & torch.isfinite(t_lps)  # [actual_n, K]
    has_any = valid_mask.any(dim=-1)  # [actual_n]
    if not has_any.any():
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    # Use a large-but-finite negative for masking to avoid 0*inf -> NaN in the
    # KL term. exp(-1e4) underflows to 0 in fp32 cleanly while still letting
    # the multiplication be exact-zero (no NaN).
    NEG_LARGE = -1e4

    safe_t_ids = torch.where(valid_mask, t_ids, torch.zeros_like(t_ids))
    t_lps_masked = torch.where(valid_mask, t_lps,
                                torch.full_like(t_lps, NEG_LARGE))
    t_dist = torch.softmax(t_lps_masked / KL_TEMP, dim=-1)  # [actual_n, K]

    s_at_topk = torch.gather(s_logits.float(), dim=-1, index=safe_t_ids)  # cast to fp32
    s_at_topk_masked = torch.where(valid_mask, s_at_topk,
                                     torch.full_like(s_at_topk, NEG_LARGE))
    s_log_dist = F.log_softmax(s_at_topk_masked / KL_TEMP, dim=-1)  # [actual_n, K]

    log_t = torch.log(t_dist.clamp_min(1e-20))
    kl_per_pos = (t_dist * (log_t - s_log_dist)).sum(dim=-1)  # [actual_n]
    kl_per_pos = kl_per_pos[has_any]
    if not torch.isfinite(kl_per_pos).all():
        # Defensive: if any remaining NaN/inf, drop those positions
        finite = torch.isfinite(kl_per_pos)
        if not finite.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0
        kl_per_pos = kl_per_pos[finite]
    return kl_per_pos.mean(), int(kl_per_pos.shape[0])


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[train-kl] model={MODEL_ID} kl_data={KL_PATH} out={OUT_DIR}")
    print(f"[train-kl] epochs={EPOCHS} lr={LR} bs={BS} grad_accum={GRAD_ACCUM}")

    records = load_records(KL_PATH, MAX_LEN)
    print(f"[train-kl] {len(records)} records")
    if not records:
        print("[train-kl] no records — abort")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    ds = PerTokenKLDataset(records, MAX_LEN)
    loader = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate)

    n_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * n_steps), n_steps)

    model.train()
    global_step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_count = 0
    accum_n_pos = 0

    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(loader):
            for ex in batch:
                input_ids = ex["input_ids"].unsqueeze(0).to(model.device)
                attn = torch.ones_like(input_ids)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[0]  # [seq, vocab]
                loss, n_pos = per_token_kl_loss(
                    logits, ex["response_start"], ex["teacher_ids"], ex["teacher_lps"]
                )
                if n_pos == 0:
                    continue
                scaled = loss / GRAD_ACCUM
                scaled.backward()
                accum_loss += float(loss.item())
                accum_count += 1
                accum_n_pos += n_pos

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1
                if global_step % 2 == 0 or global_step == 1:
                    avg = accum_loss / max(1, accum_count)
                    elapsed = time.time() - t0
                    print(f"  step {global_step}/{n_steps} ep={epoch} "
                          f"loss={avg:.4f} pos={accum_n_pos} elapsed={elapsed:.0f}s",
                          flush=True)
                accum_loss = 0.0
                accum_count = 0
                accum_n_pos = 0

        # Flush remaining
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            sched.step()
            opt.zero_grad()
            global_step += 1
            accum_loss = 0.0
            accum_count = 0
            accum_n_pos = 0

        # Save adapter at end of each epoch
        adapter_dir = Path(OUT_DIR) / f"epoch{epoch+1}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        print(f"[train-kl] saved {adapter_dir}")

    # Final adapter
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[train-kl] DONE. saved final adapter to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
