from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load your PDF document
loader = PyPDFLoader("sem.pdf")
documents = loader.load()

# Step 2: Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create a vector store from the documents
vectorstore = FAISS.from_documents(texts, embeddings)

# Step 5: Initialize the Ollama LLM with LLaMA 2 7B
llm = OllamaLLM(model="llama2")  # Updated model name to "llama2:7b"

# Step 6: Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 7: Ask a question using the `invoke` method
query = "What is syllabus of Cloud Computing?"
result = qa_chain.invoke({"query": query})  # Updated to use `invoke`

# Step 8: Print the result
print("Answer:", result["result"])
print("Source Documents:", result["source_documents"])
