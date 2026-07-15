import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import AgentState

# Load API keys from environment variables
load_dotenv()

class QualificationReport(BaseModel):
    """
    Pydantic schema representing the BANT qualification analysis for a lead.
    """
    budget: str = Field(description="Details of the lead's budget. Return 'Unknown' if not mentioned.")
    authority: str = Field(description="Evaluation of the contact's authority to make purchasing decisions. Return 'Unknown' if unclear.")
    need: str = Field(description="Description of the lead's business need for drone fleet automation.")
    timeline: str = Field(description="Implementation or testing timeline mentioned by the lead. Return 'Unknown' if not mentioned.")
    fit_score: int = Field(description="Alignment score from 0 (poor fit) to 10 (perfect fit) with FlytBase.")
    qualification_reason: str = Field(description="Detailed explanation justifying the fit score and overall BANT assessment.")
    missing_information: List[str] = Field(description="List of key pieces of information missing to complete qualification.")


def qualification_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that qualifies a parsed lead using the BANT framework
    by synthesizing the parsed lead data and the corporate research context.
    
    Why:
    - This node performs automated pre-sales qualification to determine whether the lead
      has a high potential fit before deciding to draft custom emails or request AE time.
    """
    parsed_lead = state.get("parsed_lead", {})
    research = state.get("research", {})

    # Load API keys and prepare model
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # Using temperature=0 to get deterministic, objective qualification results
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

    # Bind the structured output schema to the LLM
    structured_llm = llm.with_structured_output(QualificationReport)

    # Compile the evaluation instructions prompt using the extended research fields
    prompt = f"""You are a Sales Qualification Assistant specializing in the BANT (Budget, Authority, Need, Timeline) framework.
Your task is to analyze the parsed lead and the corporate research report to qualify the lead for FlytBase (drone fleet automation software).

Parsed Lead Data:
{parsed_lead}

Corporate Research Report:
{research}

Instructions:
1. Evaluate the lead across the BANT categories:
   - Budget: Use both the parsed lead details and 'budget_signals' from research. If no budget information exists, return 'Unknown'.
   - Authority: Infer decision-making authority from both the contact's 'designation' and the 'organization_structure' from research. Do not overestimate authority (e.g., a 'Head of Operations' may be an Influencer rather than the Final Decision Maker). Return your reasoning for the authority level in this field.
   - Need: Evaluate using 'pain_points', 'operational_priorities', and 'potential_fit' from research. Return ONLY one of 'High', 'Medium', or 'Low' instead of long paragraphs.
   - Timeline: Use the parsed lead timeline. If unavailable, return 'Unknown'.
2. Assign a Fit Score from 0 to 10 based on ALL available evidence. Explain WHY the lead received this fit score using BANT criteria (this goes in 'qualification_reason').
3. Identify missing information such as:
   - budget
   - decision maker
   - number of drones
   - deployment size
   - procurement timeline
   - technical constraints
4. Do not invent facts. Use 'Unknown' whenever required.
"""

    # Run the model to get structured report
    report = structured_llm.invoke(prompt)

    # Save validation results to state['qualification']
    if hasattr(report, "model_dump"):
        state["qualification"] = report.model_dump()
    else:
        state["qualification"] = report.dict()

    return state
