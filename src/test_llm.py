# src/test_llm.py
from models.llm_config import get_llm

def main():
    llm = get_llm("A")  # o "B"
    resp = llm.invoke("Di una frase corta que compruebe que el modelo funciona.")
    print(resp)

if __name__ == "__main__":
    main()
