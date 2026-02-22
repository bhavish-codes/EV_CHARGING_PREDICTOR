from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    demand_summary: str
    hot_zones: List[str]
    guidelines: str
    recommendations: str
    history: List[str]

def data_interpreter_node(state: AgentState):
    """Summarizes demand and identifies hot zones."""
    demand = state.get("demand_summary", "No data available.")
    # In a real app, this would use the ML model output
    summary = f"Synthesized Analysis: {demand}. Peak usage exceeds capacity by 15%."
    return {"demand_summary": summary, "hot_zones": ["Shenzhen District TAZ 559"]}

def policy_retriever_node(state: AgentState):
    """Simulates RAG by retrieving relevant planning policies."""
    # Mocked 'retrieved' guidelines from infrastructure PDF
    retrieved_text = """
    - GUIDELINE 1: For occupancy > 80%, expand station capacity by at least 25%.
    - GUIDELINE 2: Prioritize fast-charging piles in high-traffic administrative zones.
    - GUIDELINE 3: Implement dynamic pricing during peak hours (5 PM - 8 PM).
    """
    return {"guidelines": retrieved_text}

def recommendation_generator_node(state: AgentState):
    """Generates structured recommendations based on analysis and guidelines."""
    demand = state["demand_summary"]
    guidelines = state["guidelines"]
    
    recs = f"""
### Infrastructure Expansion
- Add 4 Fast-Charging (DC) piles to {state['hot_zones'][0]}.
- Increase total transformer capacity by 150kW.

### Load Balancing
- Implement a peak-hour service fee increase (+0.2 CNY/kWh) during 17:00-20:00.
- Encourage overnight charging with reduced rates.

### Reliability
- Ensure 99.9% uptime for the new charging piles via predictive maintenance.
    """
    return {"recommendations": recs}


def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("interpreter", data_interpreter_node)
    workflow.add_node("retriever", policy_retriever_node)
    workflow.add_node("generator", recommendation_generator_node)
    
    workflow.set_entry_point("interpreter")
    workflow.add_edge("interpreter", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_agent_graph()
    print("Agent graph built successfully.")
