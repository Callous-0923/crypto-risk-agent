from __future__ import annotations

from typing import Any

from openai import DefaultHttpxClient, OpenAI

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def get_http_proxy() -> str | None:
    return (
        settings.all_proxy
        or settings.https_proxy
        or settings.http_proxy
        or settings.ws_proxy
        or None
    )


def get_ws_proxy() -> str | None:
    return settings.ws_proxy or get_http_proxy()


def get_httpx_client_kwargs(*, service: str | None = None, **overrides: Any) -> dict[str, Any]:
    kwargs = dict(overrides)
    proxy = get_http_proxy()
    if proxy and "proxy" not in kwargs:
        kwargs["proxy"] = proxy
        if service:
            logger.info("Using proxy %s for %s", proxy, service)
    return kwargs


def get_openai_client_kwargs(*, service: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": settings.ark_api_key,
        "base_url": settings.ark_base_url,
    }
    proxy = get_http_proxy()
    if proxy:
        kwargs["http_client"] = DefaultHttpxClient(proxy=proxy)
        if service:
            logger.info("Using proxy %s for %s", proxy, service)
    return kwargs


def build_openai_client(*, service: str | None = None) -> OpenAI:
    return OpenAI(**get_openai_client_kwargs(service=service))
