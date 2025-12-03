# src/graph/nodes.py

from __future__ import annotations

from typing import Any, Dict, List

from graph.state import AgentState
from chains.sentiment_chain import build_sentiment_agent_chain
from tools.stats_tools import compute_sentiment_stats


# ---------- Router node ----------

def router_node(state: AgentState) -> AgentState:
    """
    Decide si vamos por 'single' o 'batch'.

    Reglas simples:
    - Si ya viene state["route"], la respetamos.
    - Si el user_input empieza con 'batch:' => modo batch.
    - Si viene state["texts"] con una lista => modo batch.
    - En cualquier otro caso => modo single.
    """

    # Si ya se decidió antes (por memoria), no tocamos
    if "route" in state:
        return state

    user_input = state.get("user_input", "") or ""
    texts: List[str] = state.get("texts") or []

    route: str = "single"

    lower = user_input.lower().strip()
    if lower.startswith("batch:"):
        route = "batch"
        if not texts:
            # Formato: batch: texto1 || texto2 || texto3
            payload = user_input[len("batch:") :].strip()
            parts = [p.strip() for p in payload.split("||") if p.strip()]
            texts = parts
    elif texts:
        route = "batch"
    else:
        route = "single"

    new_state: AgentState = {
        **state,
        "route": route,  # type: ignore
    }
    if texts:
        new_state["texts"] = texts

    return new_state


# ---------- Single analysis node ----------

def single_analysis_node(state: AgentState) -> AgentState:
    """
    Usa la cadena de análisis de sentimiento para un solo texto.
    Acumula el resultado en state["results"] (memoria a largo plazo).
    """

    user_text = state.get("user_input", "")
    if not user_text:
        raise ValueError("single_analysis_node: state['user_input'] está vacío.")

    chain = build_sentiment_agent_chain(config="A")
    out = chain.invoke({"user_text": user_text})

    current_result: Dict[str, Any] = {
        "text": user_text,
        "sentiment": out["sentiment"],
        "score": out["score"],
        "short_reason": out["short_reason"],
        "explanation": out["explanation"],
        "suggested_reply": out["suggested_reply"],
    }

    prev_results = state.get("results") or []
    new_results = prev_results + [current_result]

    new_state: AgentState = {
        **state,
        "results": new_results,
        "sentiment": out["sentiment"],
        "score": out["score"],
        "explanation": out["explanation"],
        "suggested_reply": out["suggested_reply"],
    }
    return new_state


# ---------- Batch analysis node ----------

def batch_analysis_node(state: AgentState) -> AgentState:
    """
    Analiza múltiples textos (batch).

    Toma la lista state["texts"].
    Para cada texto, usa la misma cadena de análisis individual.
    Agrega todos los resultados a state["results"] (acumulando sobre lo anterior).
    """

    texts: List[str] = state.get("texts") or []

    # Si no hay texts, intentamos parsear user_input por líneas
    if not texts:
        user_input = state.get("user_input", "") or ""
        # Cada línea no vacía se trata como un comentario
        parts = [line.strip() for line in user_input.split("\n") if line.strip()]
        texts = parts

    if not texts:
        raise ValueError("batch_analysis_node: no hay textos para analizar.")

    chain = build_sentiment_agent_chain(config="A")

    new_results: List[Dict[str, Any]] = state.get("results") or []

    for t in texts:
        out = chain.invoke({"user_text": t})
        new_results.append(
            {
                "text": t,
                "sentiment": out["sentiment"],
                "score": out["score"],
                "short_reason": out["short_reason"],
                "explanation": out["explanation"],
                "suggested_reply": out["suggested_reply"],
            }
        )

    new_state: AgentState = {
        **state,
        "texts": texts,
        "results": new_results,
    }
    return new_state


# ---------- Stats node (tool) ----------

def stats_node(state: AgentState) -> AgentState:
    """
    Node tipo 'tool': usa funciones de Python (stats_tools) para
    calcular estadísticas agregadas de los resultados guardados.
    """

    results = state.get("results") or []
    stats = compute_sentiment_stats(results)

    new_state: AgentState = {
        **state,
        "stats": stats,
    }
    return new_state


# ---------- Final output node ----------

def final_output_node(state: AgentState) -> AgentState:
    """
    Construye un mensaje final legible para el usuario, dependiendo de la ruta.
    """

    route = state.get("route", "single")
    results = state.get("results") or []
    stats = state.get("stats") or {}

    if route == "single":
        if not results:
            raise ValueError("final_output_node: no hay resultados en modo single.")
        last = results[-1]
        sentiment = last.get("sentiment")
        score = last.get("score")
        explanation = last.get("explanation")
        reply = last.get("suggested_reply")

        msg = []
        msg.append(f"🔍 Sentiment analysis (single):")
        msg.append(f"- Sentiment: **{sentiment}** (score={score:.2f})")
        msg.append("")
        msg.append("🧠 Explanation:")
        msg.append(explanation or "")
        msg.append("")
        msg.append("✉️ Suggested reply:")
        msg.append(reply or "")

        final_output = "\n".join(msg)

    else:  # batch
        total = stats.get("total", 0)
        counts = stats.get("counts", {})
        dist = stats.get("distribution", {})

        msg = []
        msg.append("📊 Batch sentiment analysis summary")
        msg.append(f"- Total texts analyzed (this session): {total}")
        msg.append("")
        msg.append("Counts:")
        for label, cnt in counts.items():
            msg.append(f"  - {label}: {cnt}")
        msg.append("")
        msg.append("Distribution:")
        for label, frac in dist.items():
            msg.append(f"  - {label}: {frac:.2f}")

        if results:
            msg.append("")
            msg.append("🔎 Example texts (first 3):")
            for r in results[:3]:
                msg.append(f"- [{r.get('sentiment')}] {r.get('text')}")

        final_output = "\n".join(msg)

    new_state: AgentState = {
        **state,
        "final_output": final_output,
    }
    return new_state
