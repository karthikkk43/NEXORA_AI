
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Amazon S3
    s3_bucket_name: str = ""

    # Amazon Bedrock
    bedrock_chat_model: str = "us.amazon.nova-2-lite-v1:0"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"

    # Embedding Configuration
    embedding_dimension: int = 1024

    # File Upload Configuration
    max_upload_bytes: int = 10485760  # 10 MB

    # PostgreSQL (Render)
    database_url: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "atlas-ai-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_ready_timeout: int = 180

    # RAG Configuration
    min_relevance: float = 0.25

    # Human-in-the-Loop
    hitl_enabled: bool = True

    # Conversation Memory
    memory_turns: int = 6

    # Rius Observability
    rius_api_key: str = ""
    rius_service_name: str = "atlas-ai"
    rius_disabled: bool = False
    rius_capture_content: bool = False


@lru_cache
def settings() -> Settings:
    return Settings()
