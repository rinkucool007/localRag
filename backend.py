import os
import pandas as pd
import requests
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from unstructured.partition.docx import partition_docx

# Function to load .docx files as a fallback if DocxLoader is not available
def load_docx(file_path):
    elements = partition_docx(file_path)
    return [{"page_content": elem.text} for elem in elements if elem.text]

# Function to load .xlsx files using pandas
def load_excel(file_path):
    documents = []
    excel_data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
    for sheet_name, sheet_data in excel_data.items():
        for _, row in sheet_data.iterrows():
            row_text = " ".join(str(cell) for cell in row if pd.notnull(cell))
            documents.append({"page_content": row_text})
    return documents

# 1. Load documents from a directory with support for PDF, DOCX, TXT, and XLSX formats
def load_documents_from_directory(directory: str):
    documents = []
    file_types = {
        "pdf": ".pdf",
        "docx": ".docx",
        "txt": ".txt",
        "xlsx": ".xlsx"
    }
    
    # Check for files in order of preference: PDF, DOCX, TXT, XLSX
    for file_type, extension in file_types.items():
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        if files:
            print(f"Loading {file_type.upper()} files only.")
            for file_name in files:
                file_path = os.path.join(directory, file_name)
                if extension == ".pdf":
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif extension == ".docx":
                    documents.extend(load_docx(file_path))
                elif extension == ".txt":
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif extension == ".xlsx":
                    documents.extend(load_excel(file_path))
            break  # Stop after loading the first available file type
    return documents

# 2. Split text into chunks for embedding and vector store
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# 3. Create embeddings using Ollama model
def create_embeddings():
    return OllamaEmbeddings()

# 4. Create a FAISS vector database from documents and embeddings
def create_vector_db(documents, embeddings):
    document_texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(document_texts)
    
    # Create a FAISS index for fast similarity search
    vector_db = FAISS(embeddings=doc_embeddings, documents=documents)
    return vector_db

# 5. Build the vector database index for all documents in the specified folder
def build_index(folder_path):
    print("Loading documents...")
    documents = load_documents_from_directory(folder_path)
    print(f"Loaded {len(documents)} documents.")
    
    if not documents:
        print("No documents loaded. Please check the folder and try again.")
        return None
    
    print("Splitting documents...")
    split_docs = split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    
    print("Creating embeddings...")
    embeddings = create_embeddings()
    vector_db = create_vector_db(split_docs, embeddings)
    print("Vector database created.")
    
    return vector_db

# 6. Query the Ollama model locally
def query_ollama(model, input_text):
    url = f"http://127.0.0.1:11434/completion"
    payload = {
        "model": model,
        "prompt": input_text,
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    return result.get("response", "Error: No response from the model.")

# 7. Search vector DB and query the LLM with the most relevant documents
def query_llm(vector_db, query):
    print("Searching vector database for relevant documents...")
    results = vector_db.similarity_search(query)
    context = " ".join([result.page_content for result in results])
    
    # Combine the context with the user query for a more informed response
    full_input = f"Context: {context}\n\nQuestion: {query}"
    
    print("Querying the Ollama Llama 3.2 model...")
    model_id = "llama-3.2"
    response = query_ollama(model_id, full_input)
    
    return response

# 8. Main function to initialize and run the RAG process
def main():
    folder_path = "./documents"  # Replace with the path to your documents
    query_text = "Your question here"  # Replace with the user's query
    
    # Build the index
    vector_db = build_index(folder_path)
    if vector_db is None:
        print("No index created. Exiting.")
        return
    
    # Query the model
    answer = query_llm(vector_db, query_text)
    print("Answer:", answer)

# Run the main function
if __name__ == "__main__":
    main()
