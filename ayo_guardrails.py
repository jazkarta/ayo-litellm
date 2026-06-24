"""
Ayo guardrail sink: prints each guardrail trigger and pushes it to Redis (list keyed by
conversation id) for LibreChat to attach to the chat record's metadata. Best-effort.
"""

import json
import os

import redis.asyncio as aioredis

_REDIS_TTL_SECONDS = 600
_redis_client = None
_redis_init_failed = False


def _get_redis():
    global _redis_client, _redis_init_failed
    if _redis_client is not None or _redis_init_failed:
        return _redis_client
    url = os.getenv("REDIS_URL")
    if not url:
        _redis_init_failed = True
        return None
    try:
        _redis_client = aioredis.from_url(url, decode_responses=True)
    except Exception:
        _redis_init_failed = True
    return _redis_client


def _conversation_id(request_data: dict):
    user = request_data.get("user")
    if isinstance(user, str) and user.startswith("{"):
        try:
            conv = json.loads(user).get("conversationId")
            if conv:
                return conv
        except Exception:
            pass
    try:
        headers = (request_data.get("proxy_server_request") or {}).get("headers") or {}
        return headers.get("x-chat-session-id")
    except Exception:
        return None


async def record_guardrail_trigger(request_data: dict, name: str, stage: str, detail) -> None:
    """Print the trigger and push it to Redis for LibreChat to pick up."""
    print(f"[GUARDRAIL TRIGGERED] name={name} stage={stage} detail={detail}", flush=True)

    conv = _conversation_id(request_data or {})
    if not conv:
        return
    client = _get_redis()
    if client is None:
        return

    key = f"ayo:guardrails:{conv}"
    entry = json.dumps({"name": name, "stage": stage, "detail": detail})
    try:
        await client.rpush(key, entry)
        await client.expire(key, _REDIS_TTL_SECONDS)
    except Exception:
        pass
