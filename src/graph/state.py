# src/graph/state.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


RouteType = Literal["single", "batch"]


class AgentState(TypedDict, total=False):
    """
    Estado compartido entre los nodos del grafo.

    total=False => todas las claves son opcionales, así no revienta si falta alguna.
    """

    # Input bruto del usuario
    user_input: str

    # Ruta decidida por el router
    route: RouteType

    # Textos para análisis batch (lista de comentarios)
    texts: List[str]

    # Resultados individuales de análisis
    # Cada dict puede contener:
    #   - "text"
    #   - "sentiment"
    #   - "score"
    #   - "short_reason"
    #   - "explanation"
    #   - "suggested_reply"
    results: List[Dict[str, Any]]

    # Estadísticas agregadas (counts, distribution, etc.)
    stats: Dict[str, Any]

    # Para el caso single, guardamos estas claves directamente
    sentiment: str
    score: float
    explanation: str
    suggested_reply: str

    # Mensaje final en texto legible para mostrar al usuario
    final_output: str
