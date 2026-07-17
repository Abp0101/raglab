# RAGLab Evidence Workbench design system

## Intent

The interface behaves like a research instrument: dense enough for inspection, calm enough for long sessions, and explicit about provenance. It avoids SaaS-dashboard conventions such as floating glass cards, oversized radii, decorative gradients, and anonymous AI imagery.

## Tokens

| Category | Tokens | Use |
| --- | --- | --- |
| Surfaces | `paper`, `paper-raised`, `ink`, `ink-soft` | Notebook canvas and instrument rail |
| Signals | `vermilion`, `cobalt`, `acid`, `moss` | Actions, framework traces, live state, success |
| Lines | `line`, `line-strong` | Evidence divisions and measurement grids |
| Type | Instrument Sans, IBM Plex Mono | Editorial hierarchy and machine data |
| Radius | 0, 2, 4 px | Controls feel manufactured rather than inflated |
| Motion | 140–480 ms, cubic instrument easing | State transitions and evidence entrance only |

## Core patterns

### Signal ribbon

A compact row of state cells for latency, framework, evidence status, cost, and local dependencies. Values use tabular mono text. Never place document or query content in the ribbon.

### Evidence rail

Ranked chunks retain document title, page, section, score provenance, and quoted text in one inspectable unit. Selection is a real button, supports keyboard focus, and cannot be conveyed by colour alone.

### Framework switch

Five equal instrument keys expose the shared pipeline choices. The active key has a filled index block, border shift, and text label so state remains visible without colour.

### Measurement table

Frameworks are columns and metrics are rows. Bars are normalized within a metric only and always display the exact observed value. Results are labelled as observations, not rankings.

## Accessibility

- All interactive elements have visible `:focus-visible` states.
- Signal colour is paired with text, iconography, or geometry.
- Minimum interactive height is 40 px; primary controls use 44 px.
- The shell collapses to horizontal navigation below 900 px and single-column evidence below 760 px.
- Motion is disabled under `prefers-reduced-motion`.
- The document structure uses landmarks, headings, labels, tables, and live regions before ARIA supplementation.

## Do / do not

| Do | Do not |
| --- | --- |
| Show source metadata beside claims | Hide evidence behind generic “sources” chips |
| Use route templates and safe operational aggregates | Render credentials, prompts, or raw metric identifiers as decoration |
| Preserve dense comparison views | Turn every number into a separate card |
| Use sharp hierarchy and ruled space | Add glow, glass blur, blob gradients, or sparkle icons |
