import os
from typing import List, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_groq import ChatGroq
from state import AgentState

# Load environment keys
load_dotenv()

class CaseStudyReport(BaseModel):
    """
    Pydantic schema representing the final selected case study details.
    """
    customer_name: str = Field(description="Name of the customer company in the case study.")
    case_study_title: str = Field(description="Title of the case study or success story.")
    industry: str = Field(description="Industry sector of the customer in the case study.")
    summary: str = Field(description="Summary of the case study success story.")
    key_results: List[str] = Field(description="List of key results and measurable outcomes achieved.")
    why_relevant: str = Field(description="Why this case study is relevant to the current lead.")
    reference_url: str = Field(description="Official reference URL for the case study.")

class CaseStudyExtraction(BaseModel):
    """
    Internal extraction schema used by the LLM to structure outputs and track the source index
    without hallucinating the reference URL.
    """
    customer_name: str = Field(description="Name of the customer company in the case study.")
    case_study_title: str = Field(description="Title of the case study or success story.")
    industry: str = Field(description="Industry of the customer in the case study.")
    summary: str = Field(description="Summary of the case study success story.")
    key_results: List[str] = Field(description="List of key results and measurable outcomes achieved. This MUST be a list of distinct strings.")
    why_relevant: str = Field(description="Explanation of why this case study matches the current lead's context.")
    selected_result_index: int = Field(description="0-based index of the search result from which this case study was extracted.")


def search_flytbase_case_studies(industry: str) -> Tuple[str, List[str]]:
    """
    Queries Tavily Search for FlytBase success stories or case studies relevant to the industry.
    Returns a formatted context string of the results and a list of source URLs.
    
    This helper isolates the search I/O to keep the node logic clean and modular.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is not set in the environment or .env file.")

    tavily_client = TavilyClient(api_key=tavily_key)

    # Priority-based search aimed at official FlytBase success stories, including specific industry terms
    search_query = (
        f"site:flytbase.com case study OR customer stories OR Anglo American OR {industry} OR "
        f"FlytBase mining case study OR FlytBase logistics case study OR FlytBase official case studies"
    )
    search_response = tavily_client.search(query=search_query, max_results=5)

    results = search_response.get("results", [])
    
    # Fallback to broader FlytBase case studies search if the industry-specific query yields no results
    if not results:
        search_query = "site:flytbase.com case study success story customer stories Anglo American"
        search_response = tavily_client.search(query=search_query, max_results=5)
        results = search_response.get("results", [])

    results_text = []
    sources = []

    # Map search results to index references
    for idx, r in enumerate(results):
        url = r.get("url", "")
        if url:
            sources.append(url)
        title = r.get("title", "Untitled")
        content = r.get("content", "")
        results_text.append(f"[Result Index {idx}]\nTitle: {title}\nURL: {url}\nContent: {content}\n")

    return "\n".join(results_text), sources


def case_study_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that finds the single most relevant FlytBase case study
    for the lead, and updates state['case_study'].
    
    Why:
    - Referencing a highly relevant case study in outreach emails drastically improves
      response rates. This node automates the matchmaking process.
    """
    parsed_lead = state.get("parsed_lead", {})
    lead_company = parsed_lead.get("company", "Unknown")
    lead_industry = parsed_lead.get("industry", "logistics")
    qualification = state.get("qualification", {})
    lead_need = qualification.get("need", "Unknown")

    # Run Tavily Search to gather potential customer stories
    search_context, sources = search_flytbase_case_studies(lead_industry)

    # If no sources are returned from search, return early with Unknown values
    if not sources:
        report = CaseStudyReport(
            customer_name="Unknown",
            case_study_title="Unknown",
            industry="Unknown",
            summary="Unknown",
            key_results=[],
            why_relevant="Unknown",
            reference_url="Unknown"
        )
        state["case_study"] = report.model_dump() if hasattr(report, "model_dump") else report.dict()
        return state

    # Load Groq API Key and initialize client
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # High fidelity structured output extraction
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

    # Bind the extraction schema (excluding reference_url directly to avoid hallucination)
    structured_llm = llm.with_structured_output(CaseStudyExtraction)

    # Compile the extraction instructions for the Enterprise Sales Enablement Agent
    prompt = f"""You are an Enterprise Sales Enablement Agent.
Your job is to identify the SINGLE best matching FlytBase customer case study or success story from the search results.

Lead Details:
- Company: {lead_company}
- Industry: {lead_industry}
- Need: {lead_need}

Search Results:
{search_context}

Instructions:
1. Select the SINGLE best matching case study based on:
   - Industry alignment
   - Operational context
   - Business problem
   - Qualification
2. Prefer official FlytBase case studies over third-party summaries.
3. If 'Anglo American' appears naturally in the search results and is the strongest match for the lead, choose it. Never hardcode Anglo American if it's not present or not the best match.
4. Summarize the selected case study and list each key result or measurable outcome achieved as a separate item in the 'key_results' list. Do not serialize the list into a single string.
5. Explain clearly WHY the chosen case study is the strongest fit (this goes in the 'why_relevant' field).
6. Never invent facts, customer names, or results. If any information is unavailable, return 'Unknown'.
7. Identify the 0-based index of the search result from which the case study was extracted (selected_result_index). It must be between 0 and {len(sources) - 1}.
8. Do NOT generate, guess, or output any source URLs in any field.
9.The industry field must represent the selected case study customer's industry, not the lead's industry.
"""

    # Run LLM structured extraction
    extraction = structured_llm.invoke(prompt)

    # Map the reference URL programmatically from the source index returned by the LLM
    ref_url = "Unknown"
    selected_idx = extraction.selected_result_index
    if 0 <= selected_idx < len(sources):
        ref_url = sources[selected_idx]
    else:
        # Fallback to the first source url if the LLM output an out-of-bounds index
        ref_url = sources[0]

    # Map the extraction results into the final CaseStudyReport schema
    report = CaseStudyReport(
        customer_name=extraction.customer_name,
        case_study_title=extraction.case_study_title,
        industry=extraction.industry,
        summary=extraction.summary,
        key_results=extraction.key_results,
        why_relevant=extraction.why_relevant,
        reference_url=ref_url
    )

    # Update state['case_study'] with the final dictionary
    if hasattr(report, "model_dump"):
        state["case_study"] = report.model_dump()
    else:
        state["case_study"] = report.dict()

    return state
