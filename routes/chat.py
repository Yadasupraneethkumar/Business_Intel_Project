from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.weaviate_vectorstore import WeaviateVectorStore, Retriever
from llm.groq_client import get_structured_completion


router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
async def chat(request: ChatRequest):
    vector_store = None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = WeaviateVectorStore(
            collection_name="FinancialDocuments",
            embedding_function=embeddings
        )

        retriever = Retriever(vector_store)

        docs = retriever.invoke(
            query=request.question,
            company=request.company,
            year=request.year,
            top_k=2
        )

        context = "\n\n---\n\n".join(
            doc.page_content.strip()[:4000]
            for doc in docs
            if doc.page_content and doc.page_content.strip()
        )

        if not context:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "available corporate reports."
                )
            }

        prompt = f"""
You are an expert financial analyst.

Use ONLY the following context from corporate financial reports
to answer the user's question.

If the context does not contain enough information, clearly state
that there is not enough available data.

Do not invent, assume, or calculate information that is not
supported by the context.

Context:
{context}

User Question:
{request.question}

Provide a concise and accurate answer.
"""

        response = get_structured_completion(
            prompt=prompt,
            response_model=ChatResponse
        )

        return {"answer": response.answer}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if vector_store is not None:
            vector_store.close()