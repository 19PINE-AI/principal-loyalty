import { useState } from 'react'

/**
 * A reusable figure-from-paper component.
 *
 *   <PaperFigure src="arxiv_fig1_manifold.png" label="Figure 1" caption="..." />
 *
 * Renders the figure with a paper-style caption block and a click-to-zoom
 * lightbox overlay for high-resolution inspection.
 */
export default function PaperFigure({ src, label, caption, maxWidth = '780px', tight = false }) {
  const [zoom, setZoom] = useState(false)
  const url = `${import.meta.env.BASE_URL}figures/${src}`
  return (
    <>
      <figure className={`bg-white border border-ink/10 rounded-xl ${tight ? 'p-3' : 'p-5'}`}>
        <button
          type="button"
          onClick={() => setZoom(true)}
          className="block w-full"
          aria-label={`Zoom ${label || 'figure'}`}>
          <img
            src={url}
            alt={caption || label || src}
            className="mx-auto block max-w-full h-auto rounded-md hover:opacity-95 cursor-zoom-in"
            style={{ maxWidth }}
          />
        </button>
        {(label || caption) && (
          <figcaption className="text-sm text-ink/65 mt-3 leading-relaxed">
            {label && <strong className="text-ink/85">{label}.</strong>}{' '}
            {caption}
          </figcaption>
        )}
      </figure>

      {zoom && (
        <div
          className="fixed inset-0 z-50 bg-ink/85 flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setZoom(false)}
          role="button"
          aria-label="Close zoom">
          <img
            src={url}
            alt={caption || label || src}
            className="max-h-full max-w-full"
            onClick={e => e.stopPropagation()}
          />
          <button
            className="absolute top-4 right-4 text-white/80 hover:text-white text-2xl"
            onClick={() => setZoom(false)}
            aria-label="Close">✕</button>
        </div>
      )}
    </>
  )
}

/** Compact "From the paper · Figure N" header for inline embeds */
export function PaperFigureHeader({ label }) {
  return (
    <div className="flex items-center gap-2 mb-2 text-xs uppercase tracking-wider font-semibold text-ink/40">
      <span className="inline-block w-6 h-px bg-ink/30" />
      From the paper · {label}
      <span className="inline-block flex-1 h-px bg-ink/15" />
    </div>
  )
}
