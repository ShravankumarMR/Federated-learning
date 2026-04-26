from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.biometric.agent import BiometricAgent
from app.agents.federated_learning.agent import FederatedLearningAgent
from app.agents.graph_fraud.agent import GraphFraudAgent
from app.agents.rag.agent import RAGAgent
from app.mlops.monitoring.metrics import AGENT_ERROR_COUNT, AGENT_LATENCY, DISPATCH_LATENCY
from app.orchestration.langgraph.state import FraudState

_logger = logging.getLogger(__name__)

_AGENT_BIOMETRIC = "biometric"
_AGENT_GRAPH = "graph_fraud"
_AGENT_FEDERATED = "federated"
_AGENT_RAG = "rag"

_SUPPORTED_AGENTS = (_AGENT_BIOMETRIC, _AGENT_GRAPH, _AGENT_FEDERATED, _AGENT_RAG)
_POLICY_VERSION = "adaptive-mfa-v1"


def _default_agents_for_change(change_type: str) -> list[str]:
    if change_type == "biometric_drift":
        return [_AGENT_BIOMETRIC, _AGENT_FEDERATED, _AGENT_RAG]
    if change_type == "graph_drift":
        return [_AGENT_GRAPH, _AGENT_FEDERATED, _AGENT_RAG]
    if change_type == "federated_drift":
        return [_AGENT_FEDERATED, _AGENT_RAG]
    if change_type == "rag_context_change":
        return [_AGENT_RAG]
    return list(_SUPPORTED_AGENTS)


def dispatcher_node(state: FraudState) -> FraudState:
    started = time.perf_counter()
    event = state.get("param_change_event", {})
    change_type = str(event.get("change_type", "mixed"))
    selected_agents = _default_agents_for_change(change_type)

    explicit_agents = event.get("affected_agents", [])
    if explicit_agents:
        explicit = [a for a in explicit_agents if a in _SUPPORTED_AGENTS]
        if explicit:
            selected_agents = explicit

    # Always include federated consensus checks for high-magnitude changes.
    if float(event.get("delta_magnitude", 0.0)) >= 0.75 and _AGENT_FEDERATED not in selected_agents:
        selected_agents.append(_AGENT_FEDERATED)

    deduped = []
    for agent in selected_agents:
        if agent not in deduped:
            deduped.append(agent)

    state["selected_agents"] = deduped
    state.setdefault("details", {})["dispatch"] = {
        "change_type": change_type,
        "selected_agents": deduped,
        "delta_magnitude": float(event.get("delta_magnitude", 0.0)),
        "source": event.get("source", "unknown"),
    }
    DISPATCH_LATENCY.observe(max(time.perf_counter() - started, 0.0))
    return state


def _run_agent(agent_name: str, state: FraudState) -> tuple[str, dict[str, Any], str | None, float]:
    started = time.perf_counter()
    result: dict[str, Any]
    error: str | None = None
    try:
        if agent_name == _AGENT_BIOMETRIC:
            result = BiometricAgent().score(state.get("biometric_payload", {}))
        elif agent_name == _AGENT_GRAPH:
            dataset = state.get("dataset", "ieee_cis")
            result = GraphFraudAgent(dataset=dataset).score(state.get("graph_payload", {}))
        elif agent_name == _AGENT_FEDERATED:
            result = FederatedLearningAgent().score(state.get("federated_payload", {}))
        elif agent_name == _AGENT_RAG:
            result = RAGAgent().retrieve(state.get("rag_query", ""))
        else:
            result = {"risk": 0.5, "signal": "unsupported_agent"}
            error = f"Unsupported agent '{agent_name}'"
    except Exception as exc:  # pragma: no cover - safety net for runtime integrations
        result = {"risk": 0.5, "signal": f"{agent_name}_fallback"}
        error = str(exc)
    latency = max(time.perf_counter() - started, 0.0)
    return agent_name, result, error, latency


def parallel_agents_node(state: FraudState) -> FraudState:
    selected = state.get("selected_agents", list(_SUPPORTED_AGENTS))
    state["agent_errors"] = {}
    state.setdefault("details", {})["agent_execution"] = {}

    if not selected:
        return state

    max_workers = min(len(selected), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_agent, name, state): name for name in selected}
        for future in as_completed(futures):
            name = futures[future]
            agent_name, result, error, latency = future.result()
            AGENT_LATENCY.labels(agent=agent_name).observe(latency)

            if agent_name == _AGENT_BIOMETRIC:
                state["biometric_result"] = result
            elif agent_name == _AGENT_GRAPH:
                state["graph_result"] = result
            elif agent_name == _AGENT_FEDERATED:
                state["federated_result"] = result
            elif agent_name == _AGENT_RAG:
                state["rag_result"] = result

            if error:
                AGENT_ERROR_COUNT.labels(agent=agent_name).inc()
                state["agent_errors"][agent_name] = error
                _logger.warning("Agent '%s' failed: %s", agent_name, error)

            state["details"]["agent_execution"][name] = {
                "latency_seconds": round(latency, 4),
                "status": "error" if error else "ok",
            }

    return state


def decision_node(state: FraudState) -> FraudState:
    b = state.get("biometric_result", {}).get("risk")
    g = state.get("graph_result", {}).get("risk")
    f = state.get("federated_result", {}).get("risk")
    rag_conf = state.get("rag_result", {}).get("confidence")
    r = None if rag_conf is None else 1.0 - float(rag_conf)

    weighted_signals: list[tuple[float, float]] = []
    for signal, weight in ((b, 0.35), (g, 0.30), (f, 0.25), (r, 0.10)):
        if signal is None:
            continue
        weighted_signals.append((float(signal), weight))

    if weighted_signals:
        numerator = sum(signal * weight for signal, weight in weighted_signals)
        denominator = sum(weight for _, weight in weighted_signals)
        combined_risk = numerator / denominator
    else:
        combined_risk = 0.5

    confidence = max(0.0, min(1.0, 1.0 - combined_risk))
    decision = "fraud" if combined_risk >= 0.55 else "legit"

    if combined_risk >= 0.80:
        adaptive_mfa = "deny_review"
    elif combined_risk >= 0.55:
        adaptive_mfa = "step_up_mfa"
    else:
        adaptive_mfa = "allow"

    state["final_decision"] = decision
    state["final_confidence"] = round(confidence, 4)
    state["risk_score"] = round(combined_risk, 4)
    state["adaptive_mfa"] = adaptive_mfa
    state["policy_version"] = _POLICY_VERSION
    state.setdefault("details", {})["signals"] = {
        "biometric_risk": None if b is None else round(float(b), 4),
        "graph_risk": None if g is None else round(float(g), 4),
        "federated_risk": None if f is None else round(float(f), 4),
        "rag_risk": None if r is None else round(float(r), 4),
        "combined_risk": round(combined_risk, 4),
    }
    return state


def explanation_node(state: FraudState) -> FraudState:
    event = state.get("param_change_event", {})
    invoked_agents = state.get("selected_agents", [])
    evidence = {
        "combined_risk": state.get("risk_score", 0.0),
        "agent_errors": state.get("agent_errors", {}),
        "dispatch": state.get("details", {}).get("dispatch", {}),
        "signals": state.get("details", {}).get("signals", {}),
    }
    summary = (
        f"Risk score {state.get('risk_score', 0.0)} led to decision "
        f"'{state.get('final_decision', 'unknown')}' with action '{state.get('adaptive_mfa', 'allow')}'."
    )
    state["explanation"] = {
        "summary": summary,
        "trigger": str(event.get("change_type", "mixed")),
        "evidence": evidence,
        "policy_threshold": 0.55,
        "policy_version": state.get("policy_version", _POLICY_VERSION),
    }
    state.setdefault("details", {})["invoked_agents"] = invoked_agents
    return state


def build_orchestration_graph():
    graph = StateGraph(FraudState)

    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("parallel_agents", parallel_agents_node)
    graph.add_node("decision", decision_node)
    graph.add_node("explanation", explanation_node)

    graph.add_edge(START, "dispatcher")
    graph.add_edge("dispatcher", "parallel_agents")
    graph.add_edge("parallel_agents", "decision")
    graph.add_edge("decision", "explanation")
    graph.add_edge("explanation", END)

    return graph.compile()
