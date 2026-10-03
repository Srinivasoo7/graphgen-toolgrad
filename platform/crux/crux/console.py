"""Operator console and QRG pages. Not an authorization surface."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from crux.branding import AGENT_PORT, ONTOLOGY_MODULE, ONTOLOGY_UI, PRODUCT, TOKEN_MODULE

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_BULLET = re.compile(r"^[-*]\s+(.*)$")


def docs_root() -> Path:
    env = os.environ.get("CRUX_DOCS_DIR")
    if env:
        return Path(env)
    packaged = Path(__file__).resolve().parent.parent / "docs" / "user"
    if packaged.is_dir():
        return packaged
    return Path.cwd() / "docs" / "user"


def pitch_path() -> Path:
    return docs_root().parent / "pitch" / "crux-pitch.html"


def _chrome(title: str, inner: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #14110e;
      --paper: #f4efe6;
      --rule: #c9bfb0;
      --accent: #b42318;
      --muted: #5c564d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.45;
    }}
    header, main, footer {{
      max-width: 52rem;
      margin: 0 auto;
      padding: 1.25rem 1.5rem;
    }}
    header {{
      border-bottom: 3px solid var(--ink);
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: baseline;
    }}
    header a {{ color: var(--ink); }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.35rem; letter-spacing: -0.02em; }}
    .kicker {{ font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .modules {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.85rem;
      margin: 1.25rem 0;
    }}
    @media (max-width: 720px) {{ .modules {{ grid-template-columns: 1fr; }} }}
    article {{
      border: 1px solid var(--ink);
      padding: 1rem 1.1rem;
      background: #fffdf8;
    }}
    article h2 {{ margin: 0 0 0.4rem; font-size: 1.05rem; }}
    article p, li {{ color: var(--muted); font-size: 0.98rem; }}
    .warn {{
      border-left: 4px solid var(--accent);
      padding: 0.65rem 0.85rem;
      margin: 1rem 0;
      background: #f8ebe8;
    }}
    nav.guides a {{
      display: block;
      color: var(--ink);
      padding: 0.35rem 0;
      border-bottom: 1px solid var(--rule);
      text-decoration: none;
    }}
    nav.guides a:hover, nav.guides a:focus {{ color: var(--accent); }}
    footer {{ border-top: 1px solid var(--rule); color: var(--muted); font-size: 0.85rem; }}
    .doc h1 {{ font-size: 1.6rem; }}
    .doc h2 {{ font-size: 1.15rem; margin-top: 1.4rem; }}
    .doc pre {{
      overflow: auto;
      background: var(--ink);
      color: var(--paper);
      padding: 0.8rem 1rem;
      font-size: 0.85rem;
    }}
    .doc code {{ font-family: ui-monospace, Consolas, monospace; }}
  </style>
</head>
<body>
  <header>
    <div>
      <p class="kicker">{html.escape(PRODUCT)}</p>
      <strong>{html.escape(title)}</strong>
    </div>
    <nav>
      <a href="/">Console</a> ·
      <a href="/docs/README">Guides</a> ·
      <a href="/pitch">Pitch</a>
    </nav>
  </header>
  <main>
{inner}
  </main>
  <footer>Agents use {html.escape(AGENT_PORT)}. Humans configure knowledge in {html.escape(ONTOLOGY_MODULE)}.</footer>
</body>
</html>
"""


def console_html() -> str:
    inner = f"""
    <p class="kicker">Operator console</p>
    <h1>{html.escape(PRODUCT)}</h1>
    <p>Two modules. One product. Access is decided in Enterprise Ontology. Compression runs only after that decision.</p>
    <div class="modules">
      <article>
        <h2>{html.escape(ONTOLOGY_MODULE)}</h2>
        <p>Human console for models, upload, Review, ontology, and permissions.</p>
        <p><a href="{html.escape(ONTOLOGY_UI)}">{html.escape(ONTOLOGY_UI)}</a></p>
      </article>
      <article>
        <h2>{html.escape(TOKEN_MODULE)}</h2>
        <p>Unpublished compressor. Agents never receive this URL. It is not a knowledge base.</p>
        <p>After authorize: <code>POST /v1/compress</code></p>
      </article>
    </div>
    <div class="warn">
      Do not point Utopia <code>chat_base_url</code> at Headroom. Extraction, embeddings, and ontology-admin stay on the model provider.
    </div>
    <h2>Guides</h2>
    <nav class="guides">
      <a href="/docs/qrg-setup">QRG — First boot</a>
      <a href="/docs/qrg-ontology">QRG — Enterprise Ontology</a>
      <a href="/docs/qrg-tokens">QRG — Token Optimization</a>
      <a href="/docs/qrg-agents">QRG — Agents</a>
      <a href="/pitch">Product pitch (What / Why / How)</a>
    </nav>
    """
    return _chrome(f"{PRODUCT} console", inner)


_SAFE_HREF = frozenset({"http", "https", ""})


def _safe_href(url: str) -> str:
    scheme = urlparse(url).scheme.lower()
    if scheme in _SAFE_HREF:
        return url
    return "#"


def _inline_md(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: f'<a href="{_safe_href(match.group(2))}">{match.group(1)}</a>',
        escaped,
    )
    return escaped


def markdown_to_html(source: str) -> str:
    lines = source.replace("\r\n", "\n").split("\n")
    parts: list[str] = []
    in_list = False
    in_code = False
    code: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_code:
                parts.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = []
                in_code = False
            else:
                if in_list:
                    parts.append("</ul>")
                    in_list = False
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            if in_list:
                parts.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            parts.append(f"<h{level}>{_inline_md(heading.group(2))}</h{level}>")
            continue
        bullet = _BULLET.match(line)
        if bullet:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{_inline_md(bullet.group(1))}</li>")
            continue
        if in_list:
            parts.append("</ul>")
            in_list = False
        if not line.strip():
            continue
        parts.append(f"<p>{_inline_md(line)}</p>")
    if in_list:
        parts.append("</ul>")
    if in_code:
        parts.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
    return "\n".join(parts)


def docs_page(slug: str) -> str | None:
    if slug != "README" and not _SLUG.match(slug):
        return None
    path = docs_root() / f"{slug}.md"
    if not path.is_file():
        return None
    source = path.read_text(encoding="utf-8")
    return _chrome(slug, f'<div class="doc">{markdown_to_html(source)}</div>')


def pitch_html() -> str | None:
    path = pitch_path()
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")
