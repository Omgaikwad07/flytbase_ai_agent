import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState

# Load API keys from environment
load_dotenv()

class AESummaryReport(BaseModel):
    """
    Pydantic schema representing the structured summary briefing for the AE.
    """
    lead_overview: str = Field(description="Summary of company, industry, designation, and business problem")
    qualification_summary: str = Field(description="Summary of Need, Budget, Authority, Timeline, and Fit Score")
    research_summary: str = Field(description="Summary of company research, operational priorities, budget signals, and recent news")
    recommended_case_study: str = Field(description="Summary of customer and why relevant")
    go_to_market_motion: str = Field(description="GTM motion: Direct AE or Partner-led")
    recommended_next_steps: List[str] = Field(description="List of 3-5 practical next actions as separate list items.")
    risks: List[str] = Field(description="List of realistic risks supported by previous nodes as separate list items.")


def summary_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that runs at the end of the graph execution to compile
    a concise briefing for the Account Executive (AE) before joining a call.
    
    Why:
    - Allows AEs to understand the prospect profile, GTM strategy, risks, and next steps in under one minute.
    """
    parsed_lead = state.get("parsed_lead", {})
    research = state.get("research", {})
    qualification = state.get("qualification", {})
    case_study = state.get("case_study", {})
    partner = state.get("partner", {})

    # Load Groq API Key and initialize client
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # Initialize model using llama-3.1-8b-instant as requested
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

    # Bind the structured output schema
    structured_llm = llm.with_structured_output(AESummaryReport)

    # Compile the prompt
    prompt = f"""You are a Sales Operations Assistant. Generate an executive briefing for the Account Executive (AE).

Lead Details:
{parsed_lead}

Research Data:
{research}

Qualification Details:
{qualification}

Case Study Recommendation:
{case_study}

Partner Recommendation:
{partner}

Instructions:
1. Compile a concise AESummaryReport. Use only the provided information. Do not invent facts.
2. recommended_next_steps: Generate 3-5 practical next actions based on the lead context.
3. risks: Generate realistic risks supported by previous nodes (e.g. unknown budget, unknown decision maker).
4. CRITICAL rule for arrays: For recommended_next_steps and risks, you must return them as a proper JSON array of strings (e.g. ["item 1", "item 2"]). Do not return them as a single concatenated string.
"""

    messages = [
        SystemMessage(content="You are a sales assistant. Return the briefing using the AESummaryReport tool. You must output lists/arrays as actual JSON arrays of separate strings. Do not serialize list fields into a single string."),
        HumanMessage(content=prompt)
    ]

    # Run LLM structured extraction
    report = structured_llm.invoke(messages)

    # Format the structured report into a beautiful, concise markdown executive briefing
    summary_parts = [
        f"**Lead Overview**: {report.lead_overview}",
        f"**Qualification Summary**: {report.qualification_summary}",
        f"**Research Summary**: {report.research_summary}",
        f"**Recommended Case Study**: {report.recommended_case_study}",
        f"**Go-To-Market Motion**: {report.go_to_market_motion}",
        "",
        "**Recommended Next Steps**:",
    ]
    for step in report.recommended_next_steps:
        summary_parts.append(f"- {step}")
    
    summary_parts.append("")
    summary_parts.append("**Risks**:")
    for risk in report.risks:
        summary_parts.append(f"- {risk}")

    # Update state['ae_summary'] with the formatted markdown string
    state["ae_summary"] = "\n".join(summary_parts)

    return state
