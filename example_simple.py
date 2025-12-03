#!/usr/bin/env python3
"""
Ejemplo SUPER simple - Solo análisis de un texto
"""

from src.models.llm_config import load_llm
from src.graph.graph_builder import create_sentiment_graph, run_graph

# 1. Cargar modelo
llm = load_llm(model_name="llama3.2")

# 2. Crear grafo
graph = create_sentiment_graph()

# 3. Analizar un texto
texto = "¡Me encanta este producto! Es increíble."
resultado = run_graph(graph, texto)

# 4. Mostrar resultado
print(f"Texto: {texto}")
print(f"Sentimiento: {resultado.get('sentiment')}")
print(f"Score: {resultado.get('sentiment_score')}")
print(f"Explicación: {resultado.get('explanation')}")
