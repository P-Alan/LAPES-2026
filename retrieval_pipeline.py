from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

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

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.6  
    }
)

relevant_docs = retriever.invoke(pergunta)

print(relevant_docs)

if relevant_docs:

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
    # Criar modelo de IA de fallback 2 IA
    search_tool= TavilySearchResults(max_results=3)

    web_search = search_tool.invoke(pergunta)

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

# Resposta da LLM
print("\n--- RESPOSTA ---")
print(result)

# Outras perguntas: 

# 1. "