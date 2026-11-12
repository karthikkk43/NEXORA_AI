
"""Privacy-first Rius observability."""

import os
import logging
from functools import wraps

from dotenv import load_dotenv

log = logging.getLogger(__name__)


def initialize():
    """Initialize Rius observability."""

    load_dotenv()

    api_key = os.getenv("RIUS_API_KEY")

    disabled = os.getenv(
        "RIUS_DISABLED", "false"
    ).lower() in ("1", "true", "yes")

    if not api_key or disabled:
        log.info("Rius observability disabled")
        return

    try:
        import rius

        rius.init(
            service_name=os.getenv(
                "RIUS_SERVICE_NAME", "atlas-ai"
            ),
            capture_content=False,
        )

        log.info("Rius observability enabled")

    except Exception:
        log.exception(
            "Rius initialization failed"
        )


def trace(name):
    """Decorator for tracing application functions."""

    def decorate(fn):
        try:
            import rius

            return rius.observe(
                name=name,
                capture_input=False,
                capture_output=False,
            )(fn)

        except ImportError:
            return fn

    return decorate
