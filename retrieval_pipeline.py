from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
import json
import datetime
from pathlib import Path
import os


load_dotenv()

# Inicializando DataBase
persistent_directory = "db/chroma_db"

# Inicializando embbending model    1 IA
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)


pergunta = "What was NVIDIA's first graphics accelerator called?"

# Inicializando a estrutura do JSON
trace = {
    "pergunta_original": pergunta,
    "agentes": [],
    "recuperacao": {"status": "pendente", "itens": []},
    "fallback_acionado": False,
    "motivo_fallback": None,
    "resposta_final": None
}


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.6  
    }
)

trace["agentes"].append("VectorDB Retriever")

relevant_docs = retriever.invoke(pergunta)

print(relevant_docs)

if relevant_docs:

    trace["recuperacao"]["status"] = "sucesso"

    for doc in relevant_docs:
        trace["recuperacao"]["itens"].append({
            "conteudo": doc.page_content,
            "fonte": doc.metadata.get("source", "desconhecido")
        })

    print(f"\nPergunta do usuario: {pergunta}")
    # Exibe resultados
    print("--- Contexto ---\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Documento {i}:\n{doc.page_content[:100]}...\n")

    combined_input = f"""Based on the following documents, please answer this question: {pergunta}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents.
    """

else:

    trace["fallback_acionado"] = True
    trace["motivo_fallback"] = "Score de similaridade abaixo do limiar (0.6)"
    trace["agentes"].append("Tavily Search")

    # Criar modelo de IA de fallback 2 IA
    search_tool= TavilySearchResults(max_results=3)

    web_search = search_tool.invoke(pergunta)

    for item in web_search:
        trace["recuperacao"]["itens"].append({
            "conteudo": item['content'],
            "fonte": item['url']
        })

    print(f"\nPergunta do usuario: {pergunta}")
    # Exibe resultados
    print("--- Contexto ---\n")

    print("--- Contexto ---\n")
    for i, item in enumerate(web_search, 1):
        print(f"=== FONTE WEB {i} ===")
        print(f"Fonte: {item['url']}")
        print(f"Conteúdo: {item['content'][:100]}...\n")

    combined_input = f"""Based on the following documents, please answer this question: {pergunta}

    Documents:
    {chr(10).join([f"- {item['content']}" for item in web_search])}

    Please provide a clear, helpful answer using only the information from these documents.
    """

# Criar o modelo de IA llama3   3 IA
model = OllamaLLM(model="llama3")

# Define o comportamento da ia
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]


result = model.invoke(messages)

trace["agentes"].append("Llama3 LLM")
trace["resposta_final"] = result

# Resposta da LLM
print("\n--- RESPOSTA ---")
print(result)

logs = "logs"

if not os.path.exists(logs):
    os.makedirs(logs)

nome_arquivo = os.path.join(logs, f"trace_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

with open(nome_arquivo, "w", encoding="utf-8") as f:
    json.dump(trace, f, indent=4, ensure_ascii=False)


# Outras perguntas: 

# 1. "