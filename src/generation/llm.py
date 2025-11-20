"""
LLM integration for response generation.
"""
from typing import List, Optional
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class LLMManager:
    """Manage LLM for RAG generation."""
    
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Document],
        prompt_template: Optional[str] = None
    ) -> dict:
        """
        Generate a response using retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: Retrieved relevant documents
            prompt_template: Optional custom prompt template
            
        Returns:
            Dictionary with response and source documents
        """
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Use custom prompt or default
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Format prompt
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Generate response
        response = self.llm.predict(formatted_prompt)
        
        # Extract source information
        sources = []
        for doc in retrieved_docs:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": response,
            "sources": sources
        }
    
    def create_qa_chain(self, retriever):
        """
        Create a RetrievalQA chain.
        
        Args:
            retriever: Document retriever
            
        Returns:
            RetrievalQA chain
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def get_llm(self):
        """
        Get the underlying LLM object.
        
        Returns:
            LLM object
        """
        return self.llm
