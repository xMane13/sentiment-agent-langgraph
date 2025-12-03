#!/usr/bin/env python3
"""
Ejemplo simple de uso del sentiment agent con Ollama
"""

from src.models.llm_config import load_llm
from src.graph.graph_builder import create_sentiment_graph, run_graph


def main():
    print("🚀 Iniciando Sentiment Agent con Ollama...\n")
    
    # Cargar el modelo Ollama
    print("📦 Cargando modelo llama3.2...")
    llm = load_llm(model_name="llama3.2", temperature=0.7)
    print("✅ Modelo cargado correctamente\n")
    
    # Crear el grafo
    print("🔨 Construyendo el grafo de análisis...")
    graph = create_sentiment_graph()
    print("✅ Grafo creado correctamente\n")
    
    # Textos de ejemplo para analizar
    textos_ejemplo = [
        "¡Me encanta este producto! Superó todas mis expectativas.",
        "Esta es la peor experiencia que he tenido. Totalmente decepcionado.",
        "El producto llegó a tiempo. Funciona como se describe.",
        "Excelente servicio al cliente! Resolvieron mi problema rápido.",
    ]
    
    print("=" * 60)
    print("ANÁLISIS DE SENTIMIENTOS")
    print("=" * 60 + "\n")
    
    for i, texto in enumerate(textos_ejemplo, 1):
        print(f"📝 Texto {i}: {texto}")
        print("-" * 60)
        
        # Ejecutar análisis
        resultado = run_graph(graph, texto)
        
        # Mostrar resultados
        print(f"😊 Sentimiento: {resultado.get('sentiment', 'N/A')}")
        print(f"📊 Score: {resultado.get('sentiment_score', 'N/A')}")
        print(f"💬 Explicación: {resultado.get('explanation', 'N/A')}")
        print(f"✉️  Respuesta: {resultado.get('reply', 'N/A')}")
        print("\n")
    
    # Análisis interactivo
    print("=" * 60)
    print("MODO INTERACTIVO")
    print("=" * 60 + "\n")
    print("Escribe un texto para analizar (o 'salir' para terminar):\n")
    
    while True:
        texto_usuario = input("👉 Tu texto: ").strip()
        
        if texto_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n👋 ¡Hasta pronto!")
            break
        
        if not texto_usuario:
            print("⚠️  Por favor escribe algo\n")
            continue
        
        print("\n🔍 Analizando...\n")
        resultado = run_graph(graph, texto_usuario)
        
        print(f"😊 Sentimiento: {resultado.get('sentiment', 'N/A')}")
        print(f"📊 Score: {resultado.get('sentiment_score', 'N/A')}")
        print(f"💬 Explicación: {resultado.get('explanation', 'N/A')}")
        print(f"✉️  Respuesta: {resultado.get('reply', 'N/A')}\n")


if __name__ == "__main__":
    main()
