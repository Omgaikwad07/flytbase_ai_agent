from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes.validation import validation_node
from nodes.parser import parser_node
from nodes.research import research_node
from nodes.qualification import qualification_node
from nodes.email import email_node
from nodes.case_study import case_study_node
from nodes.partner import partner_node
from nodes.summary import summary_node

#  Initialize the StateGraph with the AgentState structure
workflow = StateGraph(AgentState)


workflow.add_node("validation", validation_node)
workflow.add_node("parser", parser_node)
workflow.add_node("research", research_node)
workflow.add_node("qualification", qualification_node)
workflow.add_node("email", email_node)
workflow.add_node("case_study", case_study_node)
workflow.add_node("partner", partner_node)
workflow.add_node("summary", summary_node)

#  Define the edges of the graph


workflow.add_edge(START, "validation")


workflow.add_edge("validation", "parser")

workflow.add_edge("parser", "research")


workflow.add_edge("research", "qualification")

# Parallel Execution (Fan-out): Qualification triggers both Case Study and Partner concurrently
workflow.add_edge("qualification", "case_study")
workflow.add_edge("qualification", "partner")

# Join (Fan-in): Both Case Study and Partner parallel branches merge into Email Generator
workflow.add_edge("case_study", "email")
workflow.add_edge("partner", "email")


workflow.add_edge("email", "summary")

workflow.add_edge("summary", END)

# 4. Compile the graph to make it a runnable application
app = workflow.compile()
