import os
from pydoc import doc
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# === CARREGANDO DOCUMENTOS ===

def load_documents(docs_path="docs"):
    print(f"Carregando informaçoes de {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"O diretorio {docs_path} nao existe. Adicioneo aos seus arquivos.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"Sem arquivos .txt em {docs_path}. Adicione um para continuar.")

    for i, doc in enumerate(documents[:5]):  # mostra os 5 primeiros documentos
        print(f"\nDocumento {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Tamanho do arquivo: {len(doc.page_content)} characters")
        print(f"  Preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

# === EMPACOTANDO DOCUMENTOS ===

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Transformando o documento em chuncks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Tamanho: {len(chunk.page_content)} characters")
            print(f"COnteudo:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... e {len(chunks) - 5} mais chunks")
    
    return chunks

# === VETORIZANDO CHUNKS ===

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Criando embendding e armazenando no ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("--- Criando vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print("--- Terminando de criar o vector store ---")
    
    print(f"Vector store criado e salvo em {persist_directory}")
    return vectorstore

def main():
    print("main function")

    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()