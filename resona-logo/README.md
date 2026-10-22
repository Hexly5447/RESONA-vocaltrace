# resona — logo & brand identity

Professional vector identity for **resona** (ethical hacking / cybersecurity) by *coding cobras*.

Every lockup contains exactly two strings of text — `resona` and `by coding cobras`. Nothing else.

**Color system: one green + white.** All marks use `#00A651` only, with white as the background / negative space.
No navy, no teal, no silver, no gradients.

---

## Design

| Element | Rationale |
|---|---|
| **Wordmark** | Custom geometric lowercase letterforms built from circles, arcs and straight strokes on a single 100-unit x-height grid with a uniform 22% stroke weight. Circular `o`, single-storey geometric `a`, monoline `r`/`e`/`s`/`n`. Bold, confident, highly legible — drawn as pure vector paths, so it renders identically with zero font dependencies. |
| **Tagline** | `by coding cobras` in a Florentine/Renaissance-inspired italic serif — Bodoni (the original Italian modern serif) first in the stack, with Didot/Playfair/Georgia fallbacks. Centered, letterspaced, smaller scale for hierarchy. |
| **Icon** | A minimal **rising cobra** (calm, protective posture — hood spread, level head, single eye) inside a **shield**, with two **circuit-trace terminals** entering from the shield's shoulders. Defensive security + intelligence, flat vector, two-tone only. |
| **Palette** | Resona green `#00A651` + white. High contrast on white; reversed files place white marks on a solid green field. |

## Files

### Vector (SVG — master artwork)
| File | Use |
|---|---|
| `resona-logo-horizontal.svg` | **Primary lockup** — icon left, wordmark right, tagline centered beneath |
| `resona-logo-horizontal-dark.svg` | Primary lockup reversed (white on green) |
| `resona-logo-stacked.svg` | Centered stacked lockup (icon above wordmark) |
| `resona-logo-stacked-dark.svg` | Stacked lockup reversed |
| `resona-icon.svg` / `resona-icon-dark.svg` | Shield+cobra mark alone — favicon, avatar, app tile |
| `resona-wordmark.svg` | Wordmark only |
| `resona-app-icon.svg` | Rounded-square app icon (1024×1024) |

### Raster (PNG exports, `exports/`)
- `resona-logo-horizontal-light.png` / `-dark.png` — 2400 px wide
- `resona-logo-stacked-light.png` / `-dark.png` — 1400 px wide
- `resona-icon-light.png` / `-dark.png` — 1024 px
- `resona-wordmark-light.png` — 1800 px
- `resona-app-icon.png` — 1024 px

### Presentation
- `index.html` — brand board: lockups, mark, small-size behaviour, palette, typography notes.

## Usage notes

- **Clear space:** keep at least the icon's shield-width free on all sides of any lockup.
- **Minimum sizes:** horizontal lockup ≥ 180 px wide (wordmark x-height ≥ 30 px); icon ≥ 24 px.
- **Backgrounds:** light files on white surfaces, `-dark` files on solid green.
- **Don't:** recolor the green, add extra colors or gradients, re-typeset the tagline in a non-italic or non-serif face,
  stretch, outline, add shadows, or reorder the icon/wordmark.

## Re-rendering PNGs (optional)

The SVGs are the source of truth. `render.js` uses `@resvg/resvg-js` to export PNGs:

```bash
npm install
node render.js resona-logo-horizontal.svg exports/resona-logo-horizontal-light.png 2400 '#ffffff'
```
