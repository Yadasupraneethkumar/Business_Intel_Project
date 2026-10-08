import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.weaviate_vectorstore import WeaviateVectorStore
from ingestion.ingest_documents import ingest_document


router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    vector_store = None

    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported."
            )

        upload_dir = Path("data/raw_pdfs")
        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = upload_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = WeaviateVectorStore(
            collection_name="FinancialDocuments",
            embedding_function=embeddings
        )

        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store
        )

        return {
            "message": "Document uploaded and processed successfully",
            "file_name": file.filename
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if vector_store is not None:
            vector_store.close()