from typing import Tuple, Optional
from state import AgentState

def extract_email_fields(raw_email: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses a standard plain text email to extract From, Subject, and Body.
    Assumes standard headers prefixed with 'From:', 'Subject:', and 'Body:'.
    """
    sender = None
    subject = None
    body_lines = []
    in_body = False

    for line in raw_email.splitlines():
        line_stripped = line.strip()
        
        if in_body:
            body_lines.append(line)
        else:
            if line_stripped.lower().startswith("from:"):
                sender = line[len("from:"):].strip() or None
            elif line_stripped.lower().startswith("subject:"):
                subject = line[len("subject:"):].strip() or None
            elif line_stripped.lower().startswith("body:"):
                # Capture any body content on the same line as the 'Body:' header
                body_content = line[len("body:"):].strip()
                if body_content:
                    body_lines.append(body_content)
                in_body = True

    body = "\n".join(body_lines).strip() or None
    return sender, subject, body


def validation_node(state: AgentState) -> AgentState:
    """
    LangGraph workflow node to validate the incoming lead email text.
    
    Checks:
    1. Email exists (raw_email is present and not empty).
    2. Sender exists (From header).
    3. Subject exists (Subject header).
    4. Body exists (Body content).
    
    Updates state['validation'] key with:
    {
        "is_valid": bool,
        "errors": list
    }
    """
    errors = []
    
    # 1. Check if raw_email exists and is not empty
    raw_email = state.get("raw_email")
    if not raw_email or not isinstance(raw_email, str) or not raw_email.strip():
        errors.append("Email exists: Raw email is missing or empty.")
        state["validation"] = {
            "is_valid": False,
            "errors": errors
        }
        return state

    # 2. Extract fields from the standard email text format
    sender, subject, body = extract_email_fields(raw_email)

    # 3. Check for existence of each required field
    if not sender:
        errors.append("Sender exists: Email sender (From) is missing.")
    if not subject:
        errors.append("Subject exists: Email subject is missing.")
    if not body:
        errors.append("Body exists: Email body is missing.")

    # 4. Determine and save validation status
    is_valid = len(errors) == 0
    state["validation"] = {
        "is_valid": is_valid,
        "errors": errors
    }

    return state

