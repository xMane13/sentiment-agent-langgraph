from langchain_community.chat_models import ChatOllama

def get_llm(config: str = "A"):
    if config == "A":
        return ChatOllama(
            model="gemma3:1b",
            temperature=0.1,
            top_p=0.8,
            top_k=30,
        )
    elif config == "B":
        return ChatOllama(
            model="gemma3:1b",
            temperature=0.7,
            top_p=0.95,
            top_k=50,
        )
    else:
        raise ValueError(f"Unknown config: {config}")
