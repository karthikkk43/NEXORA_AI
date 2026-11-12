
import logging
import time
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.observability import trace


log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def index():
    """Return an initialized Pinecone index."""
    s = settings()

    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")

    if not s.pinecone_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is missing from .env")

    pc = Pinecone(api_key=s.pinecone_api_key)
    name = s.pinecone_index_name

    try:
        info = pc.describe_index(name)

    except Exception as exc:
        if getattr(exc, "status", None) != 404:
            raise

        log.info(
            "Creating Pinecone index %s (dimension=%d, metric=cosine)",
            name,
            s.embedding_dimension,
        )

        try:
            pc.create_index(
                name=name,
                dimension=s.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=s.pinecone_cloud,
                    region=s.pinecone_region,
                ),
            )

        except Exception as create_exc:
            if getattr(create_exc, "status", None) != 409:
                raise

        info = pc.describe_index(name)

    if int(info.dimension) != s.embedding_dimension:
        raise ValueError(
            f"Pinecone index {name} dimension={info.dimension}, "
            f"expected {s.embedding_dimension}. "
            "Use a different index name or matching EMBEDDING_DIMENSION."
        )

    if str(info.metric).lower() != "cosine":
        raise ValueError(
            f"Pinecone index {name} must use cosine metric, "
            f"got {info.metric}"
        )

    deadline = time.monotonic() + s.pinecone_ready_timeout

    while not pc.describe_index(name).status["ready"]:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index {name} was not ready within "
                f"{s.pinecone_ready_timeout}s"
            )

        time.sleep(3)

    log.info("Pinecone index %s ready", name)

    return pc.Index(name)


def namespace(tenant):
    return "tenant-" + tenant


@trace("upsert")
def upsert(tenant, records):
    return index().upsert(
        vectors=records,
        namespace=namespace(tenant),
    )


@trace("search")
def search(tenant, vector, groups, owner, limit=5):
    acl = {
        "$or": [
            {"allowed_groups": {"$in": list(groups)}},
            {"owner_sub": {"$eq": owner}},
        ]
    }

    result = index().query(
        vector=vector,
        top_k=limit,
        include_metadata=True,
        namespace=namespace(tenant),
        filter=acl,
    )

    return [
        {
            "id": m.id,
            "score": m.score,
            "metadata": m.metadata or {},
        }
        for m in result.matches
    ]


def delete_document(tenant, ids):
    if ids:
        index().delete(
            ids=ids,
            namespace=namespace(tenant),
        )
