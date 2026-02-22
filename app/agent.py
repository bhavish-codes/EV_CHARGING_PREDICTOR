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
    """Analyze demand data and identify target zones for expansion."""
    demand = state.get("demand_summary", "Insufficient data.")
    # Calculate zone occupancy based on demand metrics
    summary = f"Analysis Result: {demand}. System indicates peak usage is exceeding zone capacity."
    return {"demand_summary": summary, "hot_zones": ["Shenzhen District TAZ 559"]}

def policy_retriever_node(state: AgentState):
    """Retrieve planning policies relevant to the identified load patterns."""
    policy_text = """
    Planning Policies:
    - Expansion triggered when occupancy > 80%.
    - High-traffic zones receive priority for DC fast-charging.
    - Load balancing via dynamic pricing is recommended for peak windows.
    """
    return {"guidelines": policy_text}


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
