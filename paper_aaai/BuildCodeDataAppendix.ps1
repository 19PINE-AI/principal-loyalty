[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "AnonymousCodeDataAppendix.zip")
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$stagingRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".appendix_staging"))
$expectedStaging = [IO.Path]::GetFullPath((Join-Path $repoRoot "paper_aaai\.appendix_staging"))

if ($stagingRoot -ne $expectedStaging -or
    -not $stagingRoot.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use unexpected staging path: $stagingRoot"
}

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingRoot | Out-Null

function Copy-AppendixFile {
    param(
        [Parameter(Mandatory)][string]$SourceRelative,
        [Parameter(Mandatory)][string]$DestinationRelative
    )
    $source = [IO.Path]::GetFullPath((Join-Path $repoRoot $SourceRelative))
    if (-not $source.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source escapes repository: $SourceRelative"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required appendix file is missing: $SourceRelative"
    }
    $destination = [IO.Path]::GetFullPath((Join-Path $stagingRoot $DestinationRelative))
    if (-not $destination.StartsWith($stagingRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination escapes staging directory: $DestinationRelative"
    }
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

function Copy-AppendixTree {
    param(
        [Parameter(Mandatory)][string]$SourceRelative,
        [Parameter(Mandatory)][string]$DestinationRelative
    )
    $source = [IO.Path]::GetFullPath((Join-Path $repoRoot $SourceRelative))
    $destination = [IO.Path]::GetFullPath((Join-Path $stagingRoot $DestinationRelative))
    if (-not $source.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith($stagingRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Tree path escapes its allowed root."
    }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Required appendix directory is missing: $SourceRelative"
    }
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Copy-Item -Path (Join-Path $source "*") -Destination $destination -Recurse
}

Copy-AppendixFile "paper_aaai\APPENDIX_README.md" "README.md"
Copy-AppendixFile "paper_aaai\APPENDIX_LICENSE.txt" "LICENSE.txt"
Copy-AppendixFile "requirements.txt" "requirements.txt"
Copy-AppendixTree "items\v0" "items\train"
Copy-AppendixTree "items\v0_75" "items\heldout"

$dataFiles = @(
    "data\sft_v4.jsonl",
    "data\dpo_v4_1.jsonl",
    "data\teacher_traces_v4.jsonl",
    "data\verl_train.parquet",
    "data\verl_val.parquet",
    "data\pertoken_kl\iter1_topk.jsonl"
)
foreach ($file in $dataFiles) {
    Copy-AppendixFile $file $file
}

$sourceFiles = @(
    "src\__init__.py",
    "src\agent.py",
    "src\counterparty.py",
    "src\harness.py",
    "src\items.py",
    "src\loyalty.py",
    "src\probe.py",
    "src\reward.py",
    "src\scorer.py",
    "src\vendors.py"
)
foreach ($file in $sourceFiles) {
    Copy-AppendixFile $file $file
}

$scriptMap = [ordered]@{
    "scripts\audit_trajectories.py" = "scripts\evaluation\audit_trajectories.py"
    "scripts\run_traj_only.py" = "scripts\evaluation\run_traj_only.py"
    "scripts\multi_rollout_eval.py" = "scripts\evaluation\multi_rollout_eval.py"
    "scripts\score_only.py" = "scripts\evaluation\score_only.py"
    "scripts\generate_teacher_traces.py" = "scripts\preprocessing\generate_teacher_traces.py"
    "scripts\build_sft_dataset.py" = "scripts\preprocessing\build_sft_dataset.py"
    "scripts\build_dpo_v4_1.py" = "scripts\preprocessing\build_dpo_v4_1.py"
    "scripts\build_verl_dataset.py" = "scripts\preprocessing\build_verl_dataset.py"
    "scripts\pertoken_kl_collect.py" = "scripts\preprocessing\pertoken_kl_collect.py"
    "scripts\train_pertoken_kl.py" = "scripts\training\train_pertoken_kl.py"
    "scripts\train_qwen_sft.py" = "scripts\training\train_qwen_sft.py"
    "scripts\train_qwen_dpo.py" = "scripts\training\train_qwen_dpo.py"
    "scripts\merge_lora.py" = "scripts\training\merge_lora.py"
    "scripts\run_pertoken_kl.sh" = "scripts\training\run_pertoken_kl.sh"
    "scripts\run_dapo.sh" = "scripts\training\run_dapo.sh"
    "scripts\run_llama_pipeline.sh" = "scripts\training\run_llama_pipeline.sh"
    "scripts\paired_seed_test.py" = "scripts\analysis\paired_seed_test.py"
    "scripts\per_arm_xvendor_wilcoxon.py" = "scripts\analysis\per_arm_xvendor_wilcoxon.py"
    "scripts\heldout_per_arm_wilcoxon.py" = "scripts\analysis\heldout_per_arm_wilcoxon.py"
    "scripts\dual_judge_kappa.py" = "scripts\analysis\dual_judge_kappa.py"
    "scripts\recompute_all.py" = "scripts\analysis\recompute_all.py"
    "scripts\make_figs_arxiv.py" = "scripts\analysis\make_figs_arxiv.py"
}
foreach ($entry in $scriptMap.GetEnumerator()) {
    Copy-AppendixFile $entry.Key $entry.Value
}

$identityPatterns = @(
    "Bojie",
    "Noah",
    "Pine AI",
    "19PINE",
    "19pine",
    "2606.30383",
    "github.com/19PINE-AI"
)
$textExtensions = @(".py", ".sh", ".md", ".txt", ".json", ".jsonl")
$textFiles = Get-ChildItem -LiteralPath $stagingRoot -Recurse -File |
    Where-Object { $textExtensions -contains $_.Extension }
foreach ($pattern in $identityPatterns) {
    $hits = $textFiles | Select-String -SimpleMatch -Pattern $pattern
    if ($hits) {
        $locations = ($hits | ForEach-Object { $_.Path }) -join ", "
        throw "Anonymity check failed for '$pattern' in: $locations"
    }
}

$resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Force
}
Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $resolvedOutput
Remove-Item -LiteralPath $stagingRoot -Recurse -Force

Write-Host "Created $resolvedOutput"
