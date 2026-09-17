# geoforgood2026

## Agent sandbox

A **smolagents** crew running on Gemini via Google's `google-genai` SDK
(Vertex API key, Vertex service account, or the Gemini API), backed by
**Google Earth Engine** for compute. Forked from
[natcap/ee-agent-sandbox](../ee-agent-sandbox)'s infra.

### The crew

- **Orchestrator** — evaluates the prompt, picks whichever geospatial model
  tool(s) fit (auto-loaded from `tools/models/`), calls them (or writes ad hoc
  `ee` code directly), and reports the result.
- **Researcher** — sub-agent for a fact Earth Engine can't supply (an official
  date/status, a disputed statistic) via web search + page fetch.

A model tool computes over Earth Engine and publishes its result to a shared
**results board** (`natcap_agents/results.py`) — a map layer and/or a
stats-table row — as a side effect, rather than through the final-answer text.
The marimo app reads that board to draw the map and table. `tools/models/`
holds the project's own models (`example_forest_loss.py` is a working
template — swap it for the real 5); tools a Tool Smith agent forges land in
`tools/generated/` instead (see `forge.py`) and are auto-loaded the same way.

**Regions stay server-side.** A place name resolves via `resolve_region()`
(`natcap_agents/regions.py`) to a short `region_id` — the orchestrator never
pulls a boundary's coordinates back into its own context (a country/park
polygon can be thousands of points, which would bloat every later turn). Every
tool that takes a `region` accepts that `region_id`, or a plain
`'west,south,east,north'` bounding box — nothing bulkier.

**Dataset search has exactly two sources**: `search_full_catalog` (Google's
official EE Data Catalog) and `search_community_catalog`
([gee-community-catalog.org](https://gee-community-catalog.org/about/), mostly
`projects/sat-io/...` assets). `researcher`'s web search is for external facts
only, never for finding datasets.

### App

```bash
source .venv/bin/activate
marimo run notebooks/app.py       # read-only app view
# or: marimo edit notebooks/app.py   # to edit cells / the layout
```

Type a prompt, click **Run crew**, and watch: a rolling "thinking" log in the
side column streams each plan/step/tool-call as the crew works, while the
central map and the stats table below it fill in once the run's model tool(s)
report their layers/stats. The dashboard arrangement lives in
`notebooks/layouts/app.grid.json`.

### Setup

```bash
./setup.sh
$EDITOR .env                     # project id, EE service account, model backend/API key
source .venv/bin/activate
marimo run notebooks/app.py
```

You need a GCP project with the **Vertex AI** and **Earth Engine** APIs
enabled, the project **registered for Earth Engine**, and a service account
with `roles/earthengine.writer` + `roles/serviceusage.serviceUsageConsumer`
(add `roles/aiplatform.user` only if the models use the service account).

### Model auth (`.env`)

Earth Engine always uses the service account. The models use, in order:

- `VERTEX_API_KEY` set → Vertex **Express mode** with that key (no service
  account needed for the models).
- else `LLM_BACKEND=gemini` + `GEMINI_API_KEY` → Gemini Developer API.
- else → Vertex via the **service account** (`GOOGLE_APPLICATION_CREDENTIALS`).

### Layout

```
natcap_agents/
  config.py       # .env -> Settings
  auth.py         # bootstrap(): init EE + build the Gemini model
  models.py       # VertexAIServerModel (google-genai; Vertex key / SA / Gemini API)
  safety.py       # authorized imports for the code-executing orchestrator
  results.py      # shared board: model tools publish layers/stats here
  regions.py      # resolve_region() + the region_id registry (no bulky coords)
  project_tools.py # auto-loads tools/models/ (the project's own models)
  forge.py        # auto-loads tools/generated/ (Tool Smith output, if added)
  tools/
    catalog.py           # search_full_catalog, verify_asset (official EE catalog)
    community_catalog.py # search_community_catalog (gee-community-catalog.org)
    compute.py    # compute_region_stats, get_map_tiles — publish to the board
    web.py        # researcher's web-search + page-fetch tools (facts, not datasets)
    models/       # the project's own geospatial model tools (5, eventually)
    generated/    # tool_smith output, if you add one
  agents.py       # build_crew(), build_specialists()
notebooks/
  app.py                     # the marimo dashboard (prompt -> map/stats/thinking)
  layouts/app.grid.json      # its grid arrangement (map/table/sidebar positions)
secrets/          # git-ignored; sa-key.json here
```
