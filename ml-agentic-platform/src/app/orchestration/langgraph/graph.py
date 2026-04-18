from langgraph.graph import END, START, StateGraph

from app.agents.biometric.agent import BiometricAgent
from app.agents.federated_learning.agent import FederatedLearningAgent
from app.agents.graph_fraud.agent import GraphFraudAgent
from app.agents.rag.agent import RAGAgent
from app.orchestration.langgraph.state import FraudState


def biometric_node(state: FraudState) -> FraudState:
    agent = BiometricAgent()
    state["biometric_result"] = agent.score(state.get("biometric_payload", {}))
    return state


def graph_node(state: FraudState) -> FraudState:
    agent = GraphFraudAgent()
    state["graph_result"] = agent.score(state.get("graph_payload", {}))
    return state


def federated_node(state: FraudState) -> FraudState:
    agent = FederatedLearningAgent()
    state["federated_result"] = agent.score(state.get("federated_payload", {}))
    return state


def rag_node(state: FraudState) -> FraudState:
    agent = RAGAgent()
    state["rag_result"] = agent.retrieve(state.get("rag_query", ""))
    return state


def decision_node(state: FraudState) -> FraudState:
    b = state.get("biometric_result", {}).get("risk", 0.5)
    g = state.get("graph_result", {}).get("risk", 0.5)
    f = state.get("federated_result", {}).get("risk", 0.5)
    r = 1.0 - state.get("rag_result", {}).get("confidence", 0.5)

    combined_risk = (0.35 * b) + (0.3 * g) + (0.25 * f) + (0.1 * r)
    confidence = max(0.0, min(1.0, 1.0 - combined_risk))
    decision = "fraud" if combined_risk >= 0.55 else "legit"

    state["final_decision"] = decision
    state["final_confidence"] = round(confidence, 4)
    state["details"] = {
        "biometric": state.get("biometric_result", {}),
        "graph": state.get("graph_result", {}),
        "federated": state.get("federated_result", {}),
        "rag": state.get("rag_result", {}),
        "combined_risk": round(combined_risk, 4),
    }
    return state


def build_orchestration_graph():
    graph = StateGraph(FraudState)

    graph.add_node("biometric", biometric_node)
    graph.add_node("graph", graph_node)
    graph.add_node("federated", federated_node)
    graph.add_node("rag", rag_node)
    graph.add_node("decision", decision_node)

    graph.add_edge(START, "biometric")
    graph.add_edge("biometric", "graph")
    graph.add_edge("graph", "federated")
    graph.add_edge("federated", "rag")
    graph.add_edge("rag", "decision")
    graph.add_edge("decision", END)

    return graph.compile()
