import os
from typing import List, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_groq import ChatGroq
from state import AgentState

# Load environment variables (Tavily and Groq API keys)
load_dotenv()

class ResearchReport(BaseModel):
    """
    Pydantic schema representing the structured corporate research profile
    synthesized from web search results.
    """
    company_summary: str = Field(description="Summary of the company's background, operations, and core offerings.")
    industry: str = Field(description="The sector or industry of operation.")
    headquarters: str = Field(description="City and/or country of the company's headquarters.")
    recent_news: str = Field(description="List of recent news highlights, updates, or press releases.")
    drone_use_cases: str = Field(description="Potential or existing drone use cases suitable for this company/industry.")
    potential_fit: str = Field(description="Analysis of how FlytBase's drone automation solution fits the company's needs.")
    organization_structure: str = Field(description="Organizational structure or reporting hierarchy relevant to this lead.")
    budget_signals: str = Field(description="Public information about technology investments, capex, infrastructure spending, digital transformation initiatives, or any publicly available budget signals. If unavailable return 'Unknown'.")
    operational_priorities: str = Field(description="Operational priorities extracted from recent news, investor communications, annual reports, press releases or company announcements.")


def perform_company_search(company: str, industry: str, country: str) -> Tuple[str, List[str]]:
    """
    Queries the Tavily Search API for information about the target lead company.
    Returns a combined text block of the results and a list of source URLs.
    
    This helper isolates the I/O of web searching to keep the node logic modular.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is not set in the environment or .env file.")

    tavily_client = TavilyClient(api_key=tavily_key)
    
    # Construct a search query aimed at finding the company background, headquarters, news, structure, financial signals, and drone usage
    search_query = (
        f"{company} company overview headquarters latest news "
        f"organizational structure annual report investor relations "
        f"technology investment capex digital transformation drone use cases"
    )
    search_response = tavily_client.search(query=search_query, max_results=5)

    results_text = []
    sources = []
    
    # Process and clean search results into a clean string context for the LLM
    for result in search_response.get("results", []):
        url = result.get("url", "")
        if url:
            sources.append(url)
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        results_text.append(f"Title: {title}\nURL: {url}\nContent: {content}\n")

    return "\n".join(results_text), sources


def research_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that researches the parsed lead company.
    
    Why:
    - This node gathers external context about the lead company (such as domain,
      headquarters, recent news, and potential drone use cases) so that subsequent
      qualification and email personalization nodes have high-quality, up-to-date data.
    """
    # Extract parsed lead info containing company, industry, and country details
    parsed_lead = state.get("parsed_lead", {})
    company = parsed_lead.get("company", "")
    industry = parsed_lead.get("industry", "")
    country = parsed_lead.get("country", "")

    # If key details are missing, skip searching to prevent empty queries and save API credits
    if not company:
        state["research"] = {
            "company_summary": "No company name available in parsed lead data.",
            "industry": industry or "Unknown",
            "headquarters": country or "Unknown",
            "recent_news": "Unknown",
            "drone_use_cases": "Unknown",
            "potential_fit": "N/A",
            "organization_structure": "Unknown",
            "budget_signals": "Unknown",
            "operational_priorities": "Unknown",
            "sources": []
        }
        return state

    # Execute search query to fetch recent web context
    search_context, sources = perform_company_search(company, industry, country)

    # Initialize Groq client to synthesize the raw search results
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # Low temperature is used to ensure factual adherence to the search results
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_key
    )

    # Configure structured outputs ensuring the synthesis fits the Pydantic schema
    structured_llm = llm.with_structured_output(ResearchReport)

    # Compile synthesis instruction prompt
    prompt = f"""You are an Enterprise Research Analyst.

Your job is to research a potential enterprise customer for FlytBase.

Using ONLY the supplied search results:

General Rules:
- Never invent facts.
- Populate every field of the ResearchReport.
- If information is unavailable, return "Unknown".
- Never guess information.

Extract and populate:

• company_summary
• industry
• headquarters
• organization_structure
• budget_signals
• recent_news
• operational_priorities
• drone_use_cases
• potential_fit

Do NOT generate or guess source URLs.

Search Results:

{search_context}
"""
    # Run the model to get structured report
    report = structured_llm.invoke(prompt)

    # Convert report object to a standard dictionary
    if hasattr(report, "model_dump"):
        result = report.model_dump()
    else:
        result = report.dict()

    # Manually append the source URLs obtained directly from Tavily search
    result["sources"] = sources
    state["research"] = result

    return state
