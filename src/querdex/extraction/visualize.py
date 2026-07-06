from __future__ import annotations

import html
import json

from querdex.extraction.models import Extraction, ExtractionRun
from querdex.schemas import Section

_PALETTE = [
    "#ffd54f",
    "#a5d6a7",
    "#90caf9",
    "#f48fb1",
    "#ce93d8",
    "#ffab91",
    "#80cbc4",
    "#e6ee9c",
    "#b0bec5",
    "#fff176",
]

_STYLE = """
:root { color-scheme: light; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0; display: flex;
       background: #fafafa; color: #212121; }
main { flex: 1; padding: 24px 32px; max-width: 860px; }
aside { width: 320px; padding: 24px 16px; border-left: 1px solid #e0e0e0; background: #fff;
        height: 100vh; overflow-y: auto; position: sticky; top: 0; }
h1 { font-size: 1.2rem; } h2 { font-size: 1rem; margin-top: 1.4em; }
section.qx-section { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
                     padding: 14px 18px; margin-bottom: 12px; white-space: pre-wrap; line-height: 1.55; }
.qx-section-head { font-size: 0.75rem; color: #757575; margin-bottom: 6px; font-family: monospace; }
mark { border-radius: 3px; padding: 0 2px; cursor: help; }
mark.qx-hidden { background: transparent !important; }
.qx-legend label { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.85rem; }
.qx-swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
.qx-item { font-size: 0.8rem; border-left: 3px solid #ccc; padding: 4px 8px; margin: 6px 0;
           background: #fafafa; }
.qx-item a { color: inherit; text-decoration: none; }
.qx-attrs { color: #616161; font-family: monospace; font-size: 0.72rem; }
.qx-warn { border: 1px solid #ef9a9a; background: #ffebee; border-radius: 8px; padding: 12px 16px; }
.qx-stats { font-size: 0.8rem; color: #616161; }
"""

_SCRIPT = """
document.querySelectorAll('.qx-legend input').forEach(function (cb) {
  cb.addEventListener('change', function () {
    document.querySelectorAll('mark').forEach(function (m) {
      if (m.dataset.cls === cb.dataset.cls) {
        m.classList.toggle('qx-hidden', !cb.checked);
      }
    });
  });
});
"""


def _class_colors(extractions: list[Extraction]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for extraction in extractions:
        if extraction.extraction_class not in colors:
            colors[extraction.extraction_class] = _PALETTE[len(colors) % len(_PALETTE)]
    return colors


def _tooltip(extraction: Extraction) -> str:
    lines = [extraction.extraction_class]
    lines.extend(f"{key}: {value}" for key, value in extraction.attributes.items())
    if extraction.alignment == "fuzzy":
        lines.append("(fuzzy match)")
    return "\n".join(lines)


def _render_section(section: Section, spans: list[tuple[int, Extraction]], colors: dict[str, str]) -> str:
    """Render one section's content with <mark> highlights.

    Overlapping spans are rendered flat: a span starting inside an earlier
    highlight is skipped in the text (it remains in the sidebar list).
    """
    content = section.content
    parts: list[str] = []
    cursor = 0
    for ext_index, extraction in sorted(spans, key=lambda item: item[1].char_start or 0):
        start = extraction.char_start or 0
        end = min(extraction.char_end or start, len(content))
        if start < cursor or end <= start:
            continue
        parts.append(html.escape(content[cursor:start]))
        color = colors[extraction.extraction_class]
        parts.append(
            f'<mark id="ext-{ext_index}" style="background:{color}" '
            f'data-cls="{html.escape(extraction.extraction_class, quote=True)}" '
            f'title="{html.escape(_tooltip(extraction), quote=True)}">'
            f"{html.escape(content[start:end])}</mark>"
        )
        cursor = end
    parts.append(html.escape(content[cursor:]))
    head = f"{section.section_id} · page {section.page_number}"
    return (
        f'<section class="qx-section"><div class="qx-section-head">{html.escape(head)}</div>'
        f"{''.join(parts)}</section>"
    )


def render_extraction_html(run: ExtractionRun, sections: list[Section]) -> str:
    """Render a self-contained HTML review page for an extraction run."""
    colors = _class_colors(run.extractions)

    spans_by_section: dict[str, list[tuple[int, Extraction]]] = {}
    unaligned: list[tuple[int, Extraction]] = []
    for index, extraction in enumerate(run.extractions):
        if extraction.section_id is not None and extraction.char_start is not None:
            spans_by_section.setdefault(extraction.section_id, []).append((index, extraction))
        else:
            unaligned.append((index, extraction))

    body_parts = [_render_section(s, spans_by_section.get(s.section_id, []), colors) for s in sections]

    class_counts: dict[str, int] = {}
    for extraction in run.extractions:
        class_counts[extraction.extraction_class] = class_counts.get(extraction.extraction_class, 0) + 1
    legend_items = "".join(
        f'<label><input type="checkbox" checked data-cls="{html.escape(cls, quote=True)}">'
        f'<span class="qx-swatch" style="background:{colors[cls]}"></span>'
        f"{html.escape(cls)} ({count})</label>"
        for cls, count in sorted(class_counts.items())
    )

    sidebar_items: list[str] = []
    for index, extraction in enumerate(run.extractions):
        attrs = html.escape(json.dumps(extraction.attributes, ensure_ascii=False)) if extraction.attributes else ""
        color = colors[extraction.extraction_class]
        location = (
            f"p{extraction.page_number} · {extraction.section_id}"
            if extraction.section_id is not None
            else "ungrounded"
        )
        text = html.escape(extraction.extraction_text)
        inner = f"<b>{html.escape(extraction.extraction_class)}</b> · {html.escape(location)}<br>{text}"
        if attrs:
            inner += f'<br><span class="qx-attrs">{attrs}</span>'
        if extraction.section_id is not None:
            inner = f'<a href="#ext-{index}">{inner}</a>'
        sidebar_items.append(f'<div class="qx-item" style="border-left-color:{color}">{inner}</div>')

    warn_block = ""
    if unaligned:
        items = "".join(
            f"<li><b>{html.escape(e.extraction_class)}</b>: {html.escape(e.extraction_text)}</li>"
            for _, e in unaligned
        )
        warn_block = (
            '<div class="qx-warn"><b>Ungrounded extractions</b> - the model produced text that '
            f"could not be located in the source. Verify before trusting.<ul>{items}</ul></div>"
        )

    stats = run.stats
    stats_line = (
        f"{len(run.extractions)} extractions · {stats.chunk_count} chunks · {stats.llm_calls} LLM calls · "
        f"{stats.passes} pass(es) · exact {stats.exact_count} / fuzzy {stats.fuzzy_count} / "
        f"unaligned {stats.unaligned_count} · {stats.latency_ms} ms"
    )

    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8"><title>querdex extraction · {html.escape(run.doc_id)}</title>'
        f"<style>{_STYLE}</style></head><body>"
        f"<main><h1>Extraction run {html.escape(run.run_id)}</h1>"
        f'<p class="qx-stats">{html.escape(run.task.description)}</p>'
        f'<p class="qx-stats">{stats_line}</p>'
        f"{warn_block}{''.join(body_parts)}</main>"
        f'<aside><h2>Classes</h2><div class="qx-legend">{legend_items}</div>'
        f"<h2>Extractions</h2>{''.join(sidebar_items)}</aside>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
