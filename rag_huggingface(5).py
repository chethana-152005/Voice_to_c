#RAG Reasoning layer -integrating LLM to generte response
import os
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 1. Configuration & API Keys
# -----------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file.")

class RAGReasoningEngine:
    def __init__(self):
        print("--> Initializing RAG Engine (Groq Cloud)...")
        
        # 1. Initialize Embeddings (Local - Free)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initialize LLM (Groq Cloud)
        # FIX: Updated model name to 'llama-3.1-8b-instant' (the new supported model)
        self.llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant" 
        )
        
        self.vector_store = None
        self.rag_chain = None

    # -------------------------------------------------------------------------
    # Knowledge Sources
    # -------------------------------------------------------------------------
    def load_knowledge_base(self):
        print("--> Loading Knowledge Sources...")
        
        faq_data = [
            "Q: How do I reset my router? A: Locate the small reset button on the back and hold it for 10 seconds.",
            "Q: What is the warranty? A: All devices have a 2-year manufacturer warranty.",
            "Q: How to update software? A: Go to Settings > System > Software Update."
        ]
        kb_data = [
            "Policy ID: 101. Returns are allowed within 30 days if the package is unopened.",
            "Server Status: All systems operational."
        ]
        
        documents = [Document(page_content=text) for text in faq_data + kb_data]
        return documents

    def build_vector_index(self):
        documents = self.load_knowledge_base()
        print(f"--> Indexing {len(documents)} documents into FAISS...")
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        
        template = """
        You are a helpful assistant. Use the following context to answer the question.
        If you don't know the answer, just say you don't know.
        
        Context: {context}
        Question: {query}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
        print("--> System Ready.")

    def structure_query(self, asr_text: str) -> str:
        print(f"--> Raw ASR Input: '{asr_text}'")
        
        clean_prompt = ChatPromptTemplate.from_template(
            "Rewrite the following spoken sentence into a concise search query. Remove filler words.\nSentence: {text}\nQuery:"
        )
        chain = clean_prompt | self.llm | StrOutputParser()
        
        try:
            structured_query = chain.invoke({"text": asr_text}).strip()
        except Exception:
            structured_query = asr_text
            
        print(f"--> Structured Query: '{structured_query}'")
        return structured_query

    def process_request(self, asr_input: str):
        clean_query = self.structure_query(asr_input)
        print("--> Retrieving context and generating response...")
        response = self.rag_chain.invoke(clean_query)
        return response

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if "gsk_" not in GROQ_API_KEY:
        print("ERROR: Please paste your valid Groq API Key (starts with gsk_) in the code!")
    else:
        engine = RAGReasoningEngine()
        engine.build_vector_index()
        
        user_input = "Um, uh, hi... I want to know about the warranty thing?"
        final_response = engine.process_request(user_input)
        
        print("\n" + "="*50)
        print(f"FINAL RESPONSE:\n{final_response}")
        print("="*50)