from typing import TypedDict

class AgentState(TypedDict):

    raw_email: str

    validation: dict

    parsed_lead: dict

    research: dict

    qualification: dict

    emails: list

    case_study: dict

    partner: dict

    ae_summary: str