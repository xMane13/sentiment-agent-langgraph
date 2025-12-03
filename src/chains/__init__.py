# src/chains/__init__.py

"""
Paquete de cadenas del agente.

Por ahora solo exponemos la cadena de análisis de sentimiento.
Más adelante podremos añadir memoria y router.
"""

from .sentiment_chain import build_sentiment_agent_chain

__all__ = [
    "build_sentiment_agent_chain",
]
