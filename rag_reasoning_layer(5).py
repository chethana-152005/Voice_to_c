#RAG Reasoning Layer:integrating LLM to give generate response
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load from .env file

# -----------------------------------------------------------------------------
# 1. Configuration and Setup
# -----------------------------------------------------------------------------
class RAGReasoningEngine:
    def __init__(self, api_key: str = None):
        """
        Initializes the LLM, Embeddings, and Vector Store.
        """
        # If an API key is provided, set it as the environment variable
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Verify key is present
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API Key is missing. Please pass it during initialization or set the OPENAI_API_KEY environment variable.")

        # Initialize the LLM (Reasoning Engine)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Initialize Embeddings for Vector Store
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Vector Store (FAISS)
        self.vector_store = None
        
        # Define the RAG Chain
        self.rag_chain = None

    # -------------------------------------------------------------------------
    # Knowledge Sources: Documents, FAQ, Structured Data
    # -------------------------------------------------------------------------
    def load_knowledge_base(self):
        """
        Simulates loading data from Documents, FAQ datasets, and Structured KBs.
        """
        print("--> Loading Knowledge Sources...")
        
        # Source A: FAQ Dataset
        faq_data = [
            "Q: How do I reset my router? A: Press and hold the reset button on the back for 10 seconds.",
            "Q: What is the warranty period? A: All hardware products come with a 2-year standard warranty.",
            "Q: Can I upgrade my plan? A: Yes, navigate to Settings > Billing > Upgrade Plan."
        ]
        
        # Source B: Structured Knowledge Base / Documents
        docs_data = [
            "Document: Network Troubleshooting Guide. If the internet is slow, restart your modem. Check for loose cables.",
            "Policy: Returns are accepted within 30 days of purchase with a valid receipt. Items must be unopened.",
            "API Reference: The /login endpoint requires a JSON payload with 'username' and 'password'."
        ]

        # Convert raw text into LangChain Document objects
        documents = [Document(page_content=text) for text in faq_data + docs_data]
        return documents

    def build_vector_index(self):
        """
        Ingests documents into FAISS (Vector Database) for retrieval.
        """
        documents = self.load_knowledge_base()
        print(f"--> Indexing {len(documents)} documents into FAISS...")
        
        # Create FAISS index from documents
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Create a retriever interface
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        
        # Define the prompt template for the final generation
        template = """
        You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Keep the answer concise.

        Context: {context}
        
        Structured Query: {query}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Construct the RAG Chain using LCEL (LangChain Expression Language)
        self.rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
        print("--> System Ready.")

    # -------------------------------------------------------------------------
    # Task 1: Convert ASR output into a structured query
    # -------------------------------------------------------------------------
    def structure_query(self, asr_text: str) -> str:
        """
        Uses the LLM to clean up noisy ASR output and extract the core intent.
        """
        print(f"--> Raw ASR Input: '{asr_text}'")
        
        structure_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query optimizer. Clean the following spoken text into a concise search query. Remove filler words like 'um', 'uh', and repetitions."),
            ("human", "{text}")
        ])
        
        chain = structure_prompt | self.llm | StrOutputParser()
        structured_query = chain.invoke({"text": asr_text})
        
        print(f"--> Structured Query: '{structured_query}'")
        return structured_query

    # -------------------------------------------------------------------------
    # Main Pipeline Execution
    # -------------------------------------------------------------------------
    def process_request(self, asr_input: str):
        """
        Full pipeline: ASR -> Structured Query -> Retrieve -> Generate
        """
        # 1. Convert ASR to Structured Query
        clean_query = self.structure_query(asr_input)
        
        # 2. & 3. Retrieve and Generate
        # The retriever is embedded in the rag_chain, which automatically
        # fetches context from FAISS before sending to the LLM.
        print("--> Retrieving context and generating response...")
        response = self.rag_chain.invoke(clean_query)
        
        return response

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in .env file.")

    # 1. Instantiate the Engine with the Key
    try:
        engine = RAGReasoningEngine(api_key=api_key)
        
        # 2. Build the Knowledge Base
        engine.build_vector_index()
        
        # 3. Simulate ASR (Speech-to-Text) Input
        # Example: User stutters or speaks naturally with filler words
        user_voice_input = "Um, uh, hello... I need to know how to how to reset my router?"
        
        # 4. Process
        final_answer = engine.process_request(user_voice_input)
        
        print("\n" + "="*40)
        print(f"Final Contextual Response: {final_answer}")
        print("="*40)

        # Test with a different query
        user_voice_input_2 = "Can I uh return this thing if I opened it?"
        final_answer_2 = engine.process_request(user_voice_input_2)
        
        print("\n" + "="*40)
        print(f"Final Contextual Response: {final_answer_2}")
        print("="*40)

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Please edit the script and insert your valid OpenAI API Key in the 'API_KEY' variable.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")