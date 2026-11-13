
from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ==========================================
    # AWS CONFIGURATION
    # ==========================================

    # Used by Amazon Bedrock
    aws_region: str = "us-east-1"

    # Used only by Amazon S3
    s3_region: str = "us-east-2"

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # ==========================================
    # AMAZON S3
    # ==========================================

    s3_bucket_name: str = "atlas-ai-43"

    max_upload_bytes: int = 10 * 1024 * 1024

    # ==========================================
    # AMAZON BEDROCK
    # ==========================================

    bedrock_chat_model: str = (
        "us.amazon.nova-2-lite-v1:0"
    )

    bedrock_embed_model: str = (
        "amazon.titan-embed-text-v2:0"
    )

    embedding_dimension: int = 1024

    # ==========================================
    # PINECONE VECTOR DATABASE
    # ==========================================

    pinecone_api_key: str = ""

    pinecone_index_name: str = "atlas-ai-index"

    pinecone_cloud: str = "aws"

    pinecone_region: str = "us-east-1"

    pinecone_ready_timeout: int = 120

    min_relevance: float = 0.25

    # ==========================================
    # POSTGRESQL DATABASE
    # ==========================================

    database_url: str = ""

    # ==========================================
    # HUMAN IN THE LOOP
    # ==========================================

    hitl_enabled: bool = True

    # ==========================================
    # CONVERSATION MEMORY
    # ==========================================

    memory_turns: int = 6

    # ==========================================
    # RIUS OBSERVABILITY
    # ==========================================

    rius_api_key: str = ""

    rius_service_name: str = "nexora-ai"

    rius_disabled: bool = False

    rius_capture_content: bool = False


@lru_cache
def settings() -> Settings:
    return Settings()
