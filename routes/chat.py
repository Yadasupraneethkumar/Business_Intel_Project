from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.chroma_vectorstore import ChromaVectorStore, Retriever
from llm.groq_client import get_structured_completion

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize local ChromaDB
        vector_store = ChromaVectorStore(
            collection_name="financial_documents",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

        # Initialize retriever
        retriever = Retriever(vector_store.client)

        # Retrieve relevant context
        docs = retriever.invoke(
            query=request.question,
            company=request.company,
            year=request.year,
            top_k=5
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
            if doc.page_content
        )

        if not context.strip():
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "available corporate reports."
                )
            }

        # Build prompt
        prompt = f"""
You are an expert financial analyst.

Use ONLY the following context from corporate financial reports
to answer the user's question.

If the context does not contain enough information to answer
the question, clearly state that there is not enough available
data. Do not invent or assume financial information.

Context:
{context}

User Question:
{request.question}

Provide a concise and accurate answer.
"""

        # Generate answer using Groq
        response_model = type(
            "ChatResponse",
            (BaseModel,),
            {
                "__annotations__": {
                    "answer": str
                }
            }
        )

        response = get_structured_completion(
            prompt=prompt,
            response_model=response_model
        )

        return {
            "answer": response.answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )