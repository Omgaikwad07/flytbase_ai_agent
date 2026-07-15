import os
from typing import List, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_groq import ChatGroq
from state import AgentState

# Load API keys from environment
load_dotenv()

class PartnerReport(BaseModel):
    """
    Pydantic schema representing the final partner and GTM motion recommendation.
    """
    go_to_market_motion: str = Field(description="Must be 'Direct AE Engagement' or 'Partner-led Motion'.")
    partner_name: str = Field(description="Name of the recommended partner. If Direct AE, return 'None'.")
    partner_region: str = Field(description="Region of the partner. If Direct AE, return 'N/A'.")
    partner_type: str = Field(description="Type of the partner. If Direct AE, return 'None'.")
    recommendation_reason: str = Field(description="Detailed reason for recommending this GTM motion and partner.")
    reference_url: str = Field(description="Official reference URL for the partner or GTM motion details.")

class PartnerExtraction(BaseModel):
    """
    Internal extraction schema used by the LLM to structure outputs and track the source index
    without hallucinating the reference URL.
    """
    go_to_market_motion: str = Field(description="Must be 'Direct AE Engagement' or 'Partner-led Motion'.")
    partner_name: str = Field(description="Name of the recommended partner. If Direct AE, return 'None'.")
    partner_region: str = Field(description="Region of the partner. If Direct AE, return 'N/A'.")
    partner_type: str = Field(description="Type of the partner. If Direct AE, return 'None'.")
    recommendation_reason: str = Field(description="Detailed reason for recommending this GTM motion and partner.")
    selected_result_index: int = Field(description="0-based index of the search result from which the partner was found. If none selected or Direct AE, return -1.")


def search_flytbase_partners() -> Tuple[str, List[str]]:
    """
    Queries Tavily Search for FlytBase partners, reseller networks, and GTM motions.
    Returns a formatted context string of the results and a list of source URLs.
    
    This helper keeps the node's searching I/O modular and separate from LLM evaluation.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is not set in the environment or .env file.")

    tavily_client = TavilyClient(api_key=tavily_key)

    # Use combination of search terms matching lead requirements and general success stories
    search_query = (
    "FlytBase partner network "
    "official FlytBase partners "
    "FlytBase channel partners "
    "FlytBase authorized partners "
    "FlytBase reseller network "
    "FlytBase LATAM partner "
    "site:flytbase.com"
)
    search_response = tavily_client.search(query=search_query, max_results=5)

    results = search_response.get("results", [])
    
    # Fallback to broader FlytBase partner searches if necessary
    if not results:
        search_query = "FlytBase partner network reseller distributor network"
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


def partner_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that recommends a Go-To-Market motion and partner selection
    based on the lead country, region, industry, qualification score, and partner availability.
    
    Why:
    - Automatically route leads either to Direct AE engagement for complex enterprise deals
      or matching local partners for regional scaling.
    """
    parsed_lead = state.get("parsed_lead", {})
    lead_country = parsed_lead.get("country", "Unknown")
    lead_industry = parsed_lead.get("industry", "Unknown")
    
    qualification = state.get("qualification", {})
    fit_score = qualification.get("fit_score", 0)
    research = state.get("research", {})
    case_study = state.get("case_study", {})

    # Execute search query to find active partner details
    search_context, sources = search_flytbase_partners()

    # Load Groq API Key and initialize client
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # High fidelity structured output extraction using llama-3.1-8b-instant
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

    # Bind the extraction schema (excluding reference_url directly to avoid hallucination)
    structured_llm = llm.with_structured_output(PartnerExtraction)

    # Compile GTM decision prompt
    prompt = f"""
You are an Enterprise Go-To-Market Strategy Assistant for FlytBase.

Your job is to determine the BEST Go-To-Market (GTM) motion for this enterprise lead.

Lead Details

Company: {parsed_lead.get("company", "Unknown")}
Country: {lead_country}
Industry: {lead_industry}
Qualification Fit Score: {fit_score}

Research Report:
{research}

Selected Case Study:
{case_study}

Search Results:
{search_context}

--------------------------------------------------

Your Tasks

1. Analyze ONLY the provided search results.

2. Decide ONE Go-To-Market motion:

- Direct AE Engagement
OR
- Partner-led Motion

3. Base your decision using:

- Lead geography
- Country
- Region
- Industry
- Enterprise complexity
- Qualification Fit Score
- Partner availability from search results

--------------------------------------------------

Decision Rules

Choose Direct AE Engagement when:

- No suitable regional partner exists.
- The account appears to be a strategic enterprise account.
- The qualification score is high.
- Enterprise complexity is high.
- Partner information is insufficient.

Choose Partner-led Motion when:

- A regional implementation partner clearly exists.
- A local partner would accelerate deployment.
- Geography favors partner-led sales.
- The search results explicitly identify an appropriate partner.

--------------------------------------------------

Partner Selection Rules

If Partner-led Motion is selected:

- Select ONLY a real partner from the Search Results.
- Never return FlytBase as the partner.
- FlytBase is the software vendor, NOT the partner.
- Partner names must come directly from the search results.
- Return:
    partner_name
    partner_region
    partner_type
    recommendation_reason
    selected_result_index

--------------------------------------------------

Direct AE Rules

If Direct AE Engagement is selected:

partner_name = "None"

partner_region = "N/A"

partner_type = "None"

selected_result_index = -1

Explain clearly why Direct AE is preferred.

--------------------------------------------------

Important Constraints

- NEVER invent partner companies.
- NEVER invent regions.
- NEVER invent partner types.
- NEVER invent URLs.
- NEVER use FlytBase itself as the partner.
- Use ONLY information present in the search results.

If no valid partner exists in the search results,
choose Direct AE Engagement.

"""

    # Run LLM structured extraction
    extraction = structured_llm.invoke(prompt)

    # Map the reference URL programmatically from the source index returned by the LLM
    ref_url = "N/A"
    selected_idx = extraction.selected_result_index
    if extraction.go_to_market_motion == "Partner-led Motion" and sources:
        if 0 <= selected_idx < len(sources):
            ref_url = sources[selected_idx]
        elif len(sources) > 0:
            # Fallback to the first source url if index is out-of-bounds
            ref_url = sources[0]

    # Map the extraction results into the final PartnerReport schema
    report = PartnerReport(
        go_to_market_motion=extraction.go_to_market_motion,
        partner_name=extraction.partner_name,
        partner_region=extraction.partner_region,
        partner_type=extraction.partner_type,
        recommendation_reason=extraction.recommendation_reason,
        reference_url=ref_url
    )

    # Update state['partner'] with the final dictionary
    if hasattr(report, "model_dump"):
        state["partner"] = report.model_dump()
    else:
        state["partner"] = report.dict()

    return state
