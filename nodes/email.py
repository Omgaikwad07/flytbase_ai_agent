import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import AgentState

# Load API keys from environment variables
load_dotenv()

class EmailSequenceExtraction(BaseModel):
    """
    Internal extraction schema used by the LLM to structure outputs.
    """
    email_1_subject: str
    email_1_body: str
    email_2_subject: str
    email_2_body: str
    email_3_subject: str
    email_3_body: str
    sequence_strategy: str

class EmailSequenceReport(BaseModel):
    """
    Final representation stored in the graph state.
    """
    email_1_subject: str
    email_1_body: str
    email_2_subject: str
    email_2_body: str
    email_3_subject: str
    email_3_body: str
    sequence_strategy: str


def determine_buyer_tone(designation: str) -> str:
    """
    Programmatically determines the email tone based on the buyer's seniority or job role.
    
    Why:
    - High-level decision makers respond better to concise executive highlights, whereas
      managers appreciate a consultative approach and engineers expect concrete technical details.
    """
    role = designation.lower()
    if any(title in role for title in ["head", "director", "vp", "cxo", "chief", "president", "founder", "lead"]):
        return "Professional, Executive, and Concise"
    elif "manager" in role:
        return "Consultative"
    elif any(title in role for title in ["engineer", "developer", "architect", "scientist", "programmer", "technical"]):
        return "Technical"
    return "Professional"


def email_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that generates an adaptive sequence of three enterprise outreach
    emails using case study evidence, partner details, and missing qualification parameters.
    
    Why:
    - Highly contextual sales sequences dramatically increase engagement over generic automated templates.
    """
    parsed_lead = state.get("parsed_lead", {})
    research = state.get("research", {})
    qualification = state.get("qualification", {})
    case_study = state.get("case_study", {})
    partner = state.get("partner", {})

    # 1. Determine the appropriate tone based on job seniority/role
    buyer_tone = determine_buyer_tone(parsed_lead.get("designation", ""))

    # 2. Extract missing information fields from the qualification agent to formulate discovery questions
    missing_info_fields = qualification.get("missing_information", [])

    # 3. Load API keys and prepare LLM
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # High fidelity structured output extraction using llama-3.3-70b-versatile for stability
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_key
    )

    # Bind the extraction schema
    structured_llm = llm.with_structured_output(EmailSequenceExtraction)

    # Compile the generation instructions prompt
    prompt = f"""Generate a progressive sequence.

Email 1 should establish relevance.

Email 2 should build credibility.

Email 3 should encourage action.

Each email should naturally continue from the previous one.

Avoid repeating the same content.
Lead Name: {parsed_lead.get("name", "Unknown")}
Designation: {parsed_lead.get("designation", "Unknown")}
Company: {parsed_lead.get("company", "Unknown")}
Industry: {parsed_lead.get("industry", "Unknown")}

Target Tone: {buyer_tone}
Missing info to ask (if any): {missing_info_fields}
Research Report:{research}
FlytBase Case Study to reference: {case_study}
GTM GTM Recommendation: {partner}

Instructions:
1. Email 1: Intro, mention research, mention Case Study, ask discovery questions ONLY about missing qualification fields, soft CTA.
2. Email 2: Follow-up, add value, expand on benefits, address company operational priorities, continue qualification.
3. Email 3: Short executive follow-up, Demo OR 15-minute discussion OR Discovery call
Choose whichever best fits the buyer context.
4. GTM Rule: If GTM is 'Partner-led Motion', mention that regional partner '{partner.get("partner_name")}' can support them. If GTM is 'Direct AE Engagement', do NOT mention partners.
5. Do not invent facts or metrics.
6. Provide sequence_strategy.
7. Formatting Rule: Represent newlines inside email bodies with escaped newlines '\\n'. Do not use literal newlines inside JSON string values.
"""

    # Run LLM structured extraction
    extraction = structured_llm.invoke(prompt)

    # Map the extraction results into the final EmailSequenceReport schema
    report = EmailSequenceReport(
        email_1_subject=extraction.email_1_subject,
        email_1_body=extraction.email_1_body,
        email_2_subject=extraction.email_2_subject,
        email_2_body=extraction.email_2_body,
        email_3_subject=extraction.email_3_subject,
        email_3_body=extraction.email_3_body,
        sequence_strategy=extraction.sequence_strategy
    )

    # Update state['emails'] with the final report list (satisfies state schema 'emails: list')
    state["emails"] = [report.model_dump() if hasattr(report, "model_dump") else report.dict()]

    return state
