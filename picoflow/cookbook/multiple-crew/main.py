import json
from datetime import date
from pathlib import Path

from picoflow.core import flow, State, llm

DSN = "llm+openai://api.siliconflow.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?api_key_env=OPENAI_API_KEY"


# ======================
# Tool / Evidence (fake Serper)
# ======================

def fake_serper_search(query: str, k: int = 8) -> dict:
    today = date.today().isoformat()
    organic = [
        {"title": "Agent frameworks overview", "link": "https://example.com/a", "snippet": "Orchestration patterns and tool use..."},
        {"title": "Structured output and evals", "link": "https://example.com/b", "snippet": "Reliability via schemas and tests..."},
        {"title": "Multi-agent collaboration", "link": "https://example.com/c", "snippet": "Debate/review/refine workflows..."},
        {"title": "Enterprise adoption patterns", "link": "https://example.com/d", "snippet": "Governance, observability, ROI..."},
    ]
    return {"query": query, "date": today, "organic": organic[:k]}


# ======================
# Flows
# ======================

@flow
def prepare_inputs(ctx: State) -> State:
    topic = ctx.data.get("topic") or ctx.input
    year = ctx.data.get("year") or date.today().year
    return ctx.update(
        input=f"topic: {topic}\nyear: {year}",
        data={**ctx.data, "topic": topic, "year": year},
    )

@flow
def search(ctx: State) -> State:
    topic = ctx.data["topic"]
    year = ctx.data["year"]
    query = f"latest AI developments {topic} {year}"
    results = fake_serper_search(query, k=8)

    lines = []
    for i, it in enumerate(results.get("organic", []), 1):
        lines.append(f"[{i}] {it['title']}\n- {it['snippet']}\n- {it['link']}")

    return ctx.update(
        evidence=ctx.evidence + ["Search results:\n" + "\n".join(lines)],
        data={**ctx.data, "search_query": query, "search_raw": results},
    )


# ======================
# Researchers (3 roles)
# ======================

researcher_a = llm(
    llm_adapter=DSN,
    stream=False,
    prompt_template=(
        "Role: Senior Data Researcher (Products & Companies)\n"
        "Inputs:\n{input}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Return EXACTLY 8 bullets (markdown list), focusing on products/companies/industry moves."
    ),
)

researcher_b = llm(
    llm_adapter=DSN,
    stream=False,
    prompt_template=(
        "Role: Senior Data Researcher (Models & Techniques)\n"
        "Inputs:\n{input}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Return EXACTLY 8 bullets (markdown list), focusing on models/techniques/benchmarks."
    ),
)

researcher_c = llm(
    llm_adapter=DSN,
    stream=False,
    prompt_template=(
        "Role: Senior Data Researcher (Evals & Best Practices)\n"
        "Inputs:\n{input}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Return EXACTLY 8 bullets (markdown list), focusing on evals/reliability/safety/tooling."
    ),
)

@flow
async def fork_researchers(ctx: State) -> State:
    c1 = await researcher_a.acall(ctx)
    c2 = await researcher_b.acall(ctx)
    c3 = await researcher_c.acall(ctx)

    branches = {
        "researcher_a": {"focus": "products_companies", "bullets_md": c1.output or ""},
        "researcher_b": {"focus": "models_techniques", "bullets_md": c2.output or ""},
        "researcher_c": {"focus": "evals_best_practices", "bullets_md": c3.output or ""},
    }

    branches_json = json.dumps(branches, ensure_ascii=False)
    return ctx.update(
        input=branches_json,
        data={**ctx.data, "branches": branches},
        evidence=ctx.evidence + ["Fork outputs stored in ctx.data['branches']; branches JSON moved into ctx.input."],
    )


# ======================
# Merge to 10 bullets (LLM only reads {input}/{evidence})
# ======================

merge_to_10 = llm(
    llm_adapter=DSN,
    stream=False,
    prompt_template=(
        "Role: Lead Researcher (Merger)\n"
        "Task: Consolidate 3 researchers' bullet lists into EXACTLY 10 bullets.\n\n"
        "Branches (JSON):\n{input}\n\n"
        "Rules:\n"
        "- Deduplicate and merge overlaps\n"
        "- Prefer evidence-grounded statements\n"
        "- Each bullet: 1–2 sentences\n\n"
        "Evidence:\n{evidence}\n\n"
        "Output only the 10 bullets (markdown list)."
    ),
)

@flow
def bullets_to_input(ctx: State) -> State:
    bullets = ctx.output or ""
    return ctx.update(
        input=bullets,
        data={**ctx.data, "bullets": bullets},
    )


# ======================
# Writer -> Editor -> Writer(final)
# ======================

writer_draft = llm(
    llm_adapter=DSN,
    stream=True,
    prompt_template=(
        "Role: Reporting Analyst\n"
        "Task: Write a detailed markdown report based on the research bullets.\n\n"
        "Research bullets:\n{input}\n\n"
        "Requirements:\n"
        "- Markdown, no code fences\n"
        "- Start with a short Summary\n"
        "- Expand each bullet into a full section: ## ...\n"
        "- Include a Sources section using evidence links\n\n"
        "Evidence:\n{evidence}\n"
    ),
)

@flow
def draft_to_input(ctx: State) -> State:
    draft = ctx.output or ""
    return ctx.update(
        input=draft,
        data={**ctx.data, "draft": draft},
    )

editor_review = llm(
    llm_adapter=DSN,
    stream=True,
    prompt_template=(
        "Role: Critical Editor\n"
        "Task: Review the draft and provide:\n"
        "1) Issues (bullet list)\n"
        "2) Rewrite plan (numbered steps)\n"
        "Focus: clarity, structure, redundancy, cautious claims, source grounding.\n\n"
        "Draft:\n{input}\n\n"
        "Evidence:\n{evidence}\n"
    ),
)

@flow
def final_pack_to_input(ctx: State) -> State:
    draft = ctx.data.get("draft", "")
    feedback = ctx.output or ""
    packed = (
        "=== Draft ===\n" + draft + "\n\n"
        "=== Editor Feedback ===\n" + feedback
    )
    return ctx.update(input=packed)

writer_final = llm(
    llm_adapter=DSN,
    stream=True,
    final=True,
    prompt_template=(
        "Role: Reporting Analyst (Final)\n"
        "Task: Revise the report based on editor feedback.\n\n"
        "{input}\n\n"
        "Requirements:\n"
        "- Output final report in Markdown\n"
        "- No code fences\n"
        "- Keep Sources grounded in evidence links\n\n"
        "Evidence:\n{evidence}\n"
    ),
)

@flow
def save_report(ctx: State) -> State:
    path = Path("output/report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ctx.output or "", encoding="utf-8")
    return ctx.update(evidence=ctx.evidence + [f"Report saved: {path.as_posix()}"])


# ======================
# DSL pipeline
# ======================

pipeline = (
    prepare_inputs
    >> search
    >> fork_researchers     # ctx.input = branches_json
    >> merge_to_10          # ctx.output = 10 bullets
    >> bullets_to_input     # ctx.input = bullets
    >> writer_draft         # ctx.output = draft
    >> draft_to_input       # ctx.input = draft
    >> editor_review        # ctx.output = feedback
    >> final_pack_to_input  # ctx.input = draft+feedback
    >> writer_final         # ctx.output = final report
    >> save_report
)

if __name__ == "__main__":
    ctx = State(input="AI Agents", data={"topic": "AI Agents"})
    ctx = pipeline.run(ctx)

    print(ctx.output)
    print("\nEVIDENCE:")
    for e in ctx.evidence:
        print("-", e)
