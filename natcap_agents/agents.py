"""Assemble the natcap geospatial crew on Gemini (via google-genai).

    Orchestrator (manager CodeAgent)
      └─ Researcher (ToolCallingAgent) — external facts via web search

The orchestrator evaluates the prompt and calls whichever geospatial model
tool(s) fit (auto-loaded from `tools/models/` — see project_tools.py), plus
generic Earth Engine catalog/compute helpers, directly: no delegation needed
for that part, since it's itself a CodeAgent that can call tools or write ad
hoc `ee` code in the same step. Model tools publish their results to the
shared `results` board (map layers + stats rows) as a side effect; the marimo
app in notebooks/app.py reads that board after each run to build the map and
stats table.

Usage:
    from natcap_agents.auth import bootstrap
    from natcap_agents.agents import build_crew
    from natcap_agents import results
    bootstrap()
    crew = build_crew()
    results.reset()
    crew.run("How much forest was lost in <region> since 2015?")
"""
from __future__ import annotations

import copy
import importlib.resources

import yaml
from smolagents import CodeAgent, ToolCallingAgent

from .auth import init_earth_engine, orchestrator_model, worker_model
from .config import Settings, load_settings
from .forge import load_forged
from .project_tools import load_project_tools
from .regions import REGION_TOOLS
from .safety import AUTHORIZED_IMPORTS
from .tools.catalog import CATALOG_TOOLS
from .tools.community_catalog import COMMUNITY_CATALOG_TOOLS
from .tools.compute import COMPUTE_TOOLS
from .tools.web import WEB_TOOLS

SPECIALIST_MAX_STEPS = 5
ORCHESTRATOR_MAX_STEPS = 8
SPECIALIST_EXECUTOR_KWARGS = {"timeout_seconds": 180}
ORCHESTRATOR_EXECUTOR_KWARGS = {"timeout_seconds": 900}

# Replaces smolagents' default managed-agent task prompt, which demands "as
# much information as possible" and a verbose report — that pushes sub-agents
# into padded final_answer(...) calls. Ask for brevity instead.
CONCISE_MANAGED_TASK = (
    "You are '{{name}}'. Your manager gave you this task:\n"
    "---\n{{task}}\n---\n"
    "Complete it, then call final_answer with a CONCISE result — a short sentence or a "
    "single value, no padding or repetition."
)


def _concise(agent):
    agent.prompt_templates["managed_agent"]["task"] = CONCISE_MANAGED_TASK
    return agent


def _preload_ee(agent):
    """Make `ee` always defined in the agent's code executor, so a model that
    forgets `import ee` still works (deterministic fix for 'ee is not defined')."""
    try:
        import ee
        agent.python_executor.send_variables({"ee": ee})
    except Exception:
        pass  # ToolCallingAgents have no python_executor; ignore.
    return agent


def _merge_tools(base: list, extra: list | None) -> list:
    """Append extra tools, skipping any whose name collides with a base tool."""
    names = {t.name for t in base}
    return base + [t for t in (extra or []) if t.name not in names]


# smolagents' planning step (triggered by planning_interval) calls the model with
# ONLY its own built-in facts-survey/plan template — it never includes an agent's
# custom `instructions` (see CodeAgent.initialize_system_prompt vs
# MultiStepAgent._generate_planning_step). So the STEP 0 tool-fit gate in the
# orchestrator's `instructions` is invisible at plan time unless spliced directly
# into the planning templates too — do that here.
STEP0_DECISION_BLOCK = (
    "## 0. Can your tools answer this?\n"
    "Before the facts survey, decide: does some combination of your available tools "
    "(the model tools, plus resolve_region / verify_asset / search_full_catalog / "
    "search_community_catalog / compute_region_stats / get_map_tiles) plausibly answer "
    "this task? Judge this from what the tools are actually built to compute — don't "
    "assume you can improvise a fix with raw ee/geemap code.\n"
    "- If no tool/combination fits, AND the user's own prompt does not explicitly grant "
    "creative license (e.g. 'be creative', 'improvise'), your plan MUST be exactly: call "
    "final_answer with a short, plain negative naming what's missing. Do not plan any ad "
    "hoc ee/geemap code, and do not plan to reach for an unrelated tool anyway.\n"
    "- Only plan ad hoc ee/geemap code as a fallback if that creative license was "
    "explicitly given.\n"
    "- A meta-question (what tools/models/datasets do you have, what can you do) is NOT "
    "the no-fit case: plan to answer it from your own tool list with one final_answer "
    "call, and never as a plain negative.\n"
    "State which case applies, in one sentence, before the rest of your plan.\n"
    "\n"
)


def _orchestrator_prompt_templates() -> dict:
    """CodeAgent's default prompt templates, with STEP0_DECISION_BLOCK spliced into
    both the initial-plan and replan templates (see comment above)."""
    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
    )
    templates = copy.deepcopy(templates)
    templates["planning"]["initial_plan"] = templates["planning"]["initial_plan"].replace(
        "## 1. Facts survey", STEP0_DECISION_BLOCK + "## 1. Facts survey", 1
    )
    templates["planning"]["update_plan_post_messages"] = templates["planning"]["update_plan_post_messages"].replace(
        "## 1. Updated facts survey",
        STEP0_DECISION_BLOCK.replace("## 0. Can your tools answer this?", "## 0. Can your tools answer this? (reconsider it)")
        + "## 1. Updated facts survey",
        1,
    )
    return templates


def build_researcher(model, max_steps: int = SPECIALIST_MAX_STEPS) -> ToolCallingAgent:
    """Answers a claim/fact Earth Engine's own data can't settle, via the open web."""
    return _concise(ToolCallingAgent(
        tools=WEB_TOOLS,
        model=model,
        name="researcher",
        description=(
            "Verifies a specific claim or fills a factual gap using the open web — "
            "something Earth Engine's own data can't settle (an official date/status, a "
            "disputed statistic, a place name). Give it ONE concrete, checkable question; "
            "it returns a short answer with the source(s) it used."
        ),
        instructions=(
            "You verify facts using web_search and visit_webpage. Workflow: (1) web_search "
            "for the claim, (2) visit_webpage on the most authoritative-looking result to "
            "confirm details in context rather than trusting a search snippet alone, (3) if "
            "sources conflict, say so and report which is more authoritative.\n"
            "RULES:\n"
            "- Answer the EXACT question asked; don't wander into unrelated background.\n"
            "- Always name the source your answer rests on.\n"
            "- If you can't find a reliable answer after a couple of searches, say so "
            "plainly instead of guessing — 'not found' is a valid, useful result.\n"
            "- You have no access to Earth Engine; don't try to verify asset ids here."
        ),
        max_steps=max_steps,
    ))


def build_crew(settings: Settings | None = None, executor_type: str = "local",
               include_forged: bool = True, include_project_tools: bool = True) -> CodeAgent:
    """Return the orchestrator with its tools and the researcher attached.

    Project model tools (tools/models/) and forged tools (tools/generated/,
    saved by a Tool Smith you add) are auto-loaded and attached directly to
    the orchestrator, so a rebuilt crew uses them with no manual wiring.
    """
    settings = settings or load_settings()
    init_earth_engine(settings)  # guarantee Earth Engine is live before any tool runs

    boss_model = orchestrator_model(settings)
    hand_model = worker_model(settings)

    project_tools = load_project_tools() if include_project_tools else []
    forged = load_forged() if include_forged else []
    researcher = build_researcher(hand_model)

    base_tools = CATALOG_TOOLS + COMMUNITY_CATALOG_TOOLS + REGION_TOOLS + COMPUTE_TOOLS
    tools = _merge_tools(base_tools, project_tools)
    tools = _merge_tools(tools, forged)

    orchestrator = CodeAgent(
        tools=tools,
        model=boss_model,
        managed_agents=[researcher],
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        executor_type=executor_type,
        executor_kwargs=ORCHESTRATOR_EXECUTOR_KWARGS,
        prompt_templates=_orchestrator_prompt_templates(),
        name="orchestrator",
        description=(
            "Checks whether a geospatial question can be answered with the available model "
            "tools, and if so, runs them and reports the result; otherwise declines."
        ),
        instructions=(
            "OUTPUT FORMAT — this governs EVERY turn, with no exceptions: reply with a "
            "brief 'Thoughts:' line and then exactly ONE Python code block, using the "
            "code-block tags shown in your system prompt. Prose on its own is never a "
            "valid reply — it is a parse error that burns a step and shows the user "
            "nothing. This includes questions about yourself (what models/tools/datasets "
            "you have, what you can do): do not chat the answer, put it inside "
            "final_answer(\"...\") in a code block. If you have nothing to compute, the "
            "whole code block is a single final_answer(...) call.\n"
            "\n"
            "You have a FIXED set of tools: geospatial model tools (see their names/"
            "descriptions below), plus resolve_region, verify_asset, search_full_catalog, "
            "search_community_catalog, compute_region_stats, get_map_tiles, and the "
            "`researcher` sub-agent for external facts. `ee`/`geemap` are available directly "
            "in code, but STEP 0 below governs whether you're allowed to use them that way.\n"
            "\n"
            "STEP 0 — can your tools actually answer this? Decide this FIRST, before touching "
            "any tool:\n"
            "- Judge, from the tools' names/descriptions and what they're actually built to "
            "compute, whether some combination of them plausibly answers the question. Don't "
            "assume you can improvise around a gap with raw ee code — that requires explicit "
            "permission (next point).\n"
            "- DEFAULT (no such permission given): if no tool or combination of tools fits, "
            "stop immediately and call final_answer with a short, plain negative — one "
            "sentence naming what's missing (no matching model tool / no matching dataset). Do "
            "NOT attempt raw ee/geemap code, do not guess, do not partially answer, do not "
            "reach for a superficially-related tool just to produce something.\n"
            "- EXCEPTION: only if the user's own prompt explicitly grants creative license "
            "(e.g. it says 'be creative', 'improvise', 'do your best even without a tool for "
            "it') may you fall back to writing ad hoc `ee`/`geemap` code yourself (always "
            "`import ee` first; it's already authenticated) to attempt an answer beyond your "
            "fixed tools. Don't infer this permission from an ambiguous or merely open-ended "
            "question — it must be explicit.\n"
            "- META-QUESTIONS are not a STEP 0 refusal: if the user is asking what tools, "
            "models or datasets you have rather than asking you to compute something, just "
            "answer from your own tool list in a single final_answer(...) call — no tool "
            "calls, no 'I can't'.\n"
            "- Otherwise, if a tool/combination does fit, proceed with the workflow below.\n"
            "\n"
            "WORKFLOW (once you've confirmed a tool fits):\n"
            "1. Decide which model tool(s) fit the question. Prefer a purpose-built model tool "
            "— it's been vetted for the correct asset ids/units.\n"
            "2. If the question names a place, call resolve_region(query) FIRST and pass the "
            "region_id it returns as `region` to whichever tool needs it. If you already have "
            "a simple bounding box, pass it directly as 'west,south,east,north'. NEVER fetch a "
            "boundary's coordinates yourself (e.g. `.getInfo()` on a geometry) and pass that "
            "JSON into a tool call — a country/park polygon can be thousands of points, and "
            "that bloats every later turn's context. resolve_region keeps the geometry "
            "server-side precisely to avoid this.\n"
            "3. Need a dataset? search_full_catalog covers Google's official EE Data Catalog; "
            "if that finds nothing, try search_community_catalog (gee-community-catalog.org) "
            "for community-contributed datasets, mostly under `projects/sat-io/...`. Those are "
            "the only two dataset sources — don't ask `researcher` to find datasets.\n"
            "4. Model tools and the compute helpers above already publish their layer(s) and "
            "stat(s) to the app's map/table as a side effect — you do NOT need to render "
            "anything yourself. Just call them and read what they return.\n"
            "5. If the question needs a fact Earth Engine can't supply, delegate ONE specific "
            "question to `researcher`.\n"
            "\n"
            "RULES:\n"
            "- Never guess an asset id or band name from memory alone if you're not sure — "
            "verify_asset first.\n"
            "- Keep compute server-side; only .getInfo() small results (a scalar, a short list "
            "— never a geometry's coordinates or a large feature collection).\n"
            "- Report a concise final answer: what you ran, the key number(s), and which "
            "model/dataset produced them. Don't restate map/table details already visible "
            "in the app."
        ),
        planning_interval=4,
        max_steps=ORCHESTRATOR_MAX_STEPS,
    )
    return _preload_ee(orchestrator)


def build_specialists(settings: Settings | None = None) -> dict:
    """Return the non-orchestrator agents individually (handy for testing)."""
    settings = settings or load_settings()
    return {"researcher": build_researcher(worker_model(settings))}
