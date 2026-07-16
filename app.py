import sys
import importlib

# 1. Force reload state and node modules first to clear Streamlit stale cache
for mod in [
    "state",
    "nodes.validation",
    "nodes.parser",
    "nodes.research",
    "nodes.qualification",
    "nodes.case_study",
    "nodes.partner",
    "nodes.email",
    "nodes.summary",
]:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# 2. Apply monkeypatch to the freshly reloaded modules to prevent LangGraph concurrent update conflicts
import nodes.case_study
import nodes.partner

orig_case_study = nodes.case_study.case_study_node
def wrapped_case_study(state):
    res_state = orig_case_study(state)
    return {"case_study": res_state.get("case_study", {})}
nodes.case_study.case_study_node = wrapped_case_study

orig_partner = nodes.partner.partner_node
def wrapped_partner(state):
    res_state = orig_partner(state)
    return {"partner": res_state.get("partner", {})}
nodes.partner.partner_node = wrapped_partner

# 3. Force reload graph module now so it imports the wrapped/patched functions
if "graph" in sys.modules:
    importlib.reload(sys.modules["graph"])

import streamlit as st
import json

# Import the compiled LangGraph application
from graph import app

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="FlytBase AI Sales Assistant",
    page_icon="🛩️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("🛩️ FlytBase AI")
    st.caption("Enterprise Lead Qualification & Sales Automation")

    st.divider()

    st.subheader("Workflow")
    st.markdown(
        """
        1. Validation  
        2. Parser  
        3. Research  
        4. Qualification  
        5. Case Study ↔ Partner *(parallel)*  
        6. Email Sequence  
        7. AE Summary  
        """
    )

    st.divider()

    st.subheader("Technology Stack")
    st.markdown(
        """
        - **LangGraph** — Workflow orchestration  
        - **LangChain** — LLM integration  
        - **Groq** — Fast inference  
        - **Tavily** — Web search  
        - **Streamlit** — UI  
        """
    )

# ──────────────────────────────────────────────
# Helper: pretty-print dict/list as indented JSON
# ──────────────────────────────────────────────

def render_json(data):
    """Render a dict or list as syntax-highlighted JSON inside a code block."""
    st.code(json.dumps(data, indent=2, default=str), language="json")


def render_emails(emails_list):
    """Render the email sequence from the first entry in the emails list."""
    if not emails_list:
        st.info("No emails generated.")
        return

    seq = emails_list[0] if isinstance(emails_list, list) else emails_list

    for i in range(1, 4):
        subject = seq.get(f"email_{i}_subject", "")
        body = seq.get(f"email_{i}_body", "")
        st.markdown(f"**Email {i}**")
        st.markdown(f"**Subject:** {subject}")
        st.text(body)
        if i < 3:
            st.divider()

    strategy = seq.get("sequence_strategy", "")
    if strategy:
        st.markdown(f"**Sequence Strategy:** {strategy}")


# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────

st.title("FlytBase AI Sales Assistant")
st.markdown("**Enterprise Lead Qualification and Sales Automation**")
st.markdown("---")

email_input = st.text_area(
    "Paste Lead Email",
    height=200,
    placeholder=(
        "From: John Doe <john@company.com>\n"
        "Subject: Interested in FlytBase\n"
        "Body:\n"
        "Hi, I am John Doe, VP Operations at Acme Corp..."
    ),
)

run_button = st.button("▶  Run Workflow", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────

if run_button:
    if not email_input.strip():
        st.warning("Please paste a lead email before running the workflow.")
    else:
        # Build the initial state expected by the graph
        initial_state = {
            "raw_email": email_input,
            "validation": {},
            "parsed_lead": {},
            "research": {},
            "qualification": {},
            "emails": [],
            "case_study": {},
            "partner": {},
            "ae_summary": "",
        }

        try:
            with st.spinner("Running AI Workflow..."):
                result = app.invoke(initial_state)

            st.success("Workflow completed successfully.")

            # ── 1. Validation ──────────────────────────
            with st.expander("1 · Validation", expanded=False):
                validation = result.get("validation", {})
                is_valid = validation.get("is_valid", False)
                st.markdown(f"**Status:** {'✅ Valid' if is_valid else '❌ Invalid'}")
                errors = validation.get("errors", [])
                if errors:
                    st.markdown("**Errors:**")
                    for err in errors:
                        st.markdown(f"- {err}")

            # ── 2. Parsed Lead ─────────────────────────
            with st.expander("2 · Parsed Lead", expanded=False):
                render_json(result.get("parsed_lead", {}))

            # ── 3. Research Report ─────────────────────
            with st.expander("3 · Research Report", expanded=False):
                render_json(result.get("research", {}))

            # ── 4. Qualification ───────────────────────
            with st.expander("4 · Qualification", expanded=False):
                render_json(result.get("qualification", {}))

            # ── 5. Case Study ──────────────────────────
            with st.expander("5 · Case Study", expanded=False):
                render_json(result.get("case_study", {}))

            # ── 6. Partner Recommendation ──────────────
            with st.expander("6 · Partner Recommendation", expanded=False):
                render_json(result.get("partner", {}))

            # ── 7. Email Sequence ──────────────────────
            with st.expander("7 · Email Sequence", expanded=True):
                render_emails(result.get("emails", []))

            # ── 8. AE Summary ─────────────────────────
            with st.expander("8 · AE Summary", expanded=True):
                ae_summary = result.get("ae_summary", "")
                if ae_summary:
                    st.markdown(ae_summary)
                else:
                    st.info("No AE summary generated.")

        except Exception as exc:
            st.error(f"Workflow failed: {exc}")
