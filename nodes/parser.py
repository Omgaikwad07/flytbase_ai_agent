import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import AgentState

# Load environment variables from .env
load_dotenv()

class ParsedLead(BaseModel):
    """
    Pydantic schema representing the structured lead information extracted from the email.
    """
    name: str = Field(description="The full name of the sender/lead.")
    designation: str = Field(description="The job title or designation of the lead.")
    company: str = Field(description="The company name the lead is representing.")
    industry: str = Field(description="The industry sector of the company.")
    country: str = Field(description="The country where the company or lead is based.")
    email: str = Field(description="The contact email address of the lead.")
    pain_points: List[str] = Field(description="A list of pain points or challenges highlighted in the email.")
    timeline: str = Field(description="The expected timeline or urgency for finding/implementing a solution.")
    lead_summary: str = Field(description="A concise summary of the lead's query and their requirements.")


def parser_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node that reads the raw email from the state,
    uses a Groq model with structured output to extract lead information,
    and updates state['parsed_lead'] with the resulting dictionary.
    """
    # 1. Retrieve the raw email from the graph state
    raw_email = state.get("raw_email", "")

    # 2. Get API key from environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    # 3. Initialize the ChatGroq model with low temperature for high extraction fidelity
    # Using llama-3.1-8b-instant as it is fast, cost-effective, and fully supports structured outputs
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key
    )

    # 4. Configure the model to produce structured output adhering to the ParsedLead schema
    structured_llm = llm.with_structured_output(ParsedLead)

    # 5. Build prompt and run the extraction
    prompt = f"""You are an AI Lead Parsing Agent.

    Your task is ONLY to extract structured information.

    Do not infer information that is not present.

    If any field is missing, return an empty string.

    Extract:

    - Name
    - Designation
    - Company
    - Industry
    - Country
    - Email
    - Pain Points
    - Timeline
    - Lead Summary

    Return only structured output.

    Email:

    {raw_email}"""
    parsed_result = structured_llm.invoke(prompt)

    # 6. Convert the resulting Pydantic model to a dictionary to store in state['parsed_lead']
    if hasattr(parsed_result, "model_dump"):
        state["parsed_lead"] = parsed_result.model_dump()
    else:
        state["parsed_lead"] = parsed_result.dict()

    # 7. Return the updated state
    return state
