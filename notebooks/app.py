import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", layout_file="layouts/app.grid.json")


@app.cell
def _():
    # Make natcap_agents importable regardless of where marimo is launched from.
    import pathlib
    import sys

    _root = next(
        (p for p in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents) if (p / "natcap_agents").is_dir()),
        None,
    )
    if _root and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo

    return (mo,)


@app.cell
def _():
    # One call authenticates Earth Engine + the Gemini model from .env, and
    # builds the crew (orchestrator + geospatial model tools + researcher).
    from natcap_agents import results
    from natcap_agents.agents import build_crew
    from natcap_agents.auth import bootstrap

    setup_error = None
    settings = None
    crew = None
    try:
        settings = bootstrap()
        crew = build_crew(settings)
    except Exception as e:  # noqa: BLE001
        setup_error = f"{type(e).__name__}: {e}"
    return crew, results, settings, setup_error


@app.cell
def _(mo, settings, setup_error):
    if setup_error is not None:
        _status = mo.callout(
            mo.md(
                f"**Setup needed** — {setup_error}\n\n"
                "Copy `.env.example` to `.env` and fill in your GCP project, Earth "
                "Engine service account, and model auth, then rerun this notebook."
            ),
            kind="danger",
        )
    else:
        _status = mo.callout(
            mo.md(f"**Project:** `{settings.project_id}` · **backend:** `{settings.llm_backend}`"),
            kind="success",
        )
    mo.vstack([mo.md("## NatCap geospatial agent crew"), _status])
    return


@app.cell
def _(mo):
    prompt = mo.ui.text_area(
        placeholder="e.g. How much forest was lost in <region> since 2015?",
        rows=3,
        full_width=True,
        label="Prompt",
    )
    run_button = mo.ui.run_button(label="Run crew", kind="success")
    mo.vstack([prompt, run_button])
    return prompt, run_button


@app.cell
def _(crew, mo, prompt, results, run_button, setup_error):
    # The rolling "thinking" log: streams each planning/action/tool-call step as
    # the crew produces it (mo.output.append), so this cell's output IS the live
    # sidebar. The map/stats cells below react to `board` once this cell finishes.

    def _fmt(text, limit=400):
        text = "" if text is None else str(text)
        return text if len(text) <= limit else text[:limit] + "…"

    def _describe_step(step):
        kind = type(step).__name__
        if kind == "TaskStep":
            return f"**Task**\n\n{_fmt(step.task)}"
        if kind == "PlanningStep":
            return f"**Plan**\n\n{_fmt(step.plan, 800)}"
        if kind == "ActionStep":
            lines = [f"**Step {step.step_number}**"]
            if step.model_output:
                lines.append(_fmt(step.model_output, 500))
            for tc in step.tool_calls or []:
                lines.append(f"→ `{tc.name}({tc.arguments})`")
            if step.observations:
                lines.append(f"```\n{_fmt(step.observations, 400)}\n```")
            if step.error:
                lines.append(f"⚠️ {_fmt(step.error)}")
            return "\n\n".join(lines)
        if kind == "FinalAnswerStep":
            return f"**Final answer**\n\n{_fmt(step.output, 800)}"
        return _fmt(step)

    answer = None

    if setup_error is not None:
        mo.output.replace(mo.md("_Fix the setup error above, then rerun._"))
        board = results.current()
    elif not run_button.value:
        mo.output.replace(mo.md("_Enter a prompt and click **Run crew** to start._"))
        board = results.current()
    elif not prompt.value.strip():
        mo.output.replace(mo.md("_Type a prompt first._"))
        board = results.current()
    else:
        results.reset()
        mo.output.append(mo.md(f"### Running\n\n{prompt.value}"))
        mo.output.append(mo.md("---"))
        for step in crew.run(prompt.value, stream=True):
            mo.output.append(mo.md(_describe_step(step)))
            mo.output.append(mo.md("---"))
            if type(step).__name__ == "FinalAnswerStep":
                answer = step.output
        board = results.current()
    return (board,)


@app.cell
def _(board, mo):
    import folium

    _BASEMAP_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    _BASEMAP_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

    _m = folium.Map(location=[20, 0], zoom_start=2, tiles=_BASEMAP_URL, attr=_BASEMAP_ATTR)
    for _layer in board.layers:
        folium.TileLayer(
            tiles=_layer.tile_url, attr=_layer.attribution, name=_layer.name, overlay=True, control=True,
        ).add_to(_m)
    if board.layers:
        folium.LayerControl().add_to(_m)

    mo.Html(f'<div style="height:100%;width:100%;overflow:hidden">{_m._repr_html_()}</div>')
    return


@app.cell
def _(board, mo):
    if board.rows:
        mo.ui.table(board.rows, label="Stats", selection=None)
    else:
        mo.md("_No stats yet — run a prompt that calls a model/compute tool._")
    return


if __name__ == "__main__":
    app.run()
