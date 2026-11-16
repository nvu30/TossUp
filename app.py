import json

import streamlit as st

from agent import (MAX_ITERATIONS, MAX_PAGE_READS, MAX_SEARCHES, run_agent,
                   trace_to_markdown)

MAX_RUNS_PER_SESSION = 5      # cost cap: how many runs one visitor can start

st.set_page_config(page_title="Toss Up", page_icon="🪙")
st.title("🪙 Toss Up")
st.caption("Can't decide? Give me your options. I'll research and pick one.")

if "runs" not in st.session_state:
    st.session_state.runs = 0
if "pending" not in st.session_state:
    st.session_state.pending = None

options_text = st.text_area(
    "Your options (one per line, 2 to 4)",
    "Ichiran Ramen\nTaco Mac",
    height=100,
)
vibe = st.selectbox(
    "Vibe preset (optional)",
    ["None", "Broke college student", "Treat yourself", "Lazy night"],
)
priorities = st.text_input(
    "What matters to you?",
    "cheap, good reviews, not too far from Atlanta",
)
mock = st.checkbox("Test mode (fake search results)", value=False)

VIBES = {
    "Broke college student": "lowest price matters most, good value, student-friendly",
    "Treat yourself": "quality and experience matter most, price is secondary",
    "Lazy night": "low effort, close by, quick, comfortable",
}

# ---- Step 1: user asks. Nothing external runs yet. ----
if st.button("Decide for me", type="primary"):
    options = [o.strip()[:80] for o in options_text.splitlines() if o.strip()][:4]
    if len(options) < 2:
        st.error("Please enter at least two options.")
        st.stop()
    if st.session_state.runs >= MAX_RUNS_PER_SESSION:
        st.error(f"You've used all {MAX_RUNS_PER_SESSION} runs for this session. Refresh the page to start over.")
        st.stop()
    full_priorities = priorities.strip()[:300]
    if vibe != "None":
        full_priorities += f"; vibe: {VIBES[vibe]}"
    st.session_state.pending = {"options": options, "priorities": full_priorities, "mock": mock}

# ---- Step 2: HUMAN GATE. Shows the exact inputs; the agent starts only after Approve. ----
pending = st.session_state.pending
if pending:
    gate = st.empty()
    with gate.container():
        st.warning("Approval needed before the agent calls any outside tool.")
        st.write("The agent will send exactly this to the AI model and use these tools:")
        st.code(json.dumps({
            "options": pending["options"],
            "priorities": pending["priorities"],
            "tools": ["search_web (Tavily)", "read_page (Tavily)"],
            "limits": {"max_steps": MAX_ITERATIONS, "max_searches": MAX_SEARCHES,
                       "max_page_reads": MAX_PAGE_READS},
            "test_mode": pending["mock"],
        }, indent=2), language="json")
        c1, c2 = st.columns(2)
        approve = c1.button("Approve and run", type="primary")
        cancel = c2.button("Cancel")
    if cancel:
        st.session_state.pending = None
        st.rerun()
    if approve:
        gate.empty()                      # hide the approval box while the agent runs
        st.session_state.pending = None
        st.session_state.runs += 1

        st.subheader("Live trace")
        box = st.container()

        def show(step):
            icons = {"user_input": "👤", "thought": "💭", "tool_call": "🔧",
                     "observation": "👀", "final_answer": "✅", "stop": "🛑"}
            box.markdown(f"**{icons.get(step['type'], '')} Step {step['n']} - {step['type']}**")
            box.code(str(step["content"]), language=None)

        with st.spinner(f"Thinking (max {MAX_ITERATIONS} steps)..."):
            decision, trace = run_agent(pending["options"], pending["priorities"],
                                        mock=pending["mock"], on_step=show)

        st.success(f"Pick: {decision.get('pick')}")
        for r in decision.get("reasons", []):
            st.write(f"- {r}")
        if decision.get("missing_info") and decision["missing_info"].lower() != "none":
            st.info(f"Couldn't find: {decision['missing_info']}")
        for s in decision.get("sources", []):
            st.write(f"Source: {s}")

        md = trace_to_markdown(pending["options"], pending["priorities"], decision, trace)
        st.download_button("Download execution trace (trace.md)", md, file_name="trace.md")
        st.caption(f"Runs used this session: {st.session_state.runs} of {MAX_RUNS_PER_SESSION}")
