import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GLM_BASE_URL = "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
DEFAULT_GLM_MODEL = "gpt-5.5-2026-04-24"

_STATS: dict[str, int] = {}
_STATS_LOCK = threading.Lock()
_WARNED: set[str] = set()
_DOTENV_LOADED = False
_DOTENV_LOCK = threading.Lock()


class LLMUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0
    max_retries: int = 2
    temperature: float | None = None
    max_tokens: int = 1024
    cache_dir: str | None = None
    response_format_json: bool = True


def cfg_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return config.get(name, default) if hasattr(config, "get") else getattr(config, name, default)


def llm_enabled(config: Any, name: str, default: bool = False) -> bool:
    return bool(cfg_get(config, name, default))


def fallback_to_rules(config: Any) -> bool:
    return bool(cfg_get(config, "llm_fallback_to_rules", True))


def get_llm_config(config: Any = None) -> LLMConfig:
    _load_dotenv_once()
    api_key_env = str(cfg_get(config, "llm_api_key_env", "GLM_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise LLMUnavailable(f"Missing {api_key_env}; export it before enabling ObliFlow LLM mode.")

    cache_dir = cfg_get(config, "llm_cache_dir", os.environ.get("OBLIFLOW_LLM_CACHE_DIR", ".cache/obliflow_llm"))
    return LLMConfig(
        base_url=str(cfg_get(config, "llm_base_url", os.environ.get("GLM_BASE_URL", DEFAULT_GLM_BASE_URL))),
        model=str(cfg_get(config, "llm_model", os.environ.get("GLM_MODEL", DEFAULT_GLM_MODEL))),
        api_key=api_key,
        timeout_s=float(cfg_get(config, "llm_timeout_s", 30.0)),
        max_retries=int(cfg_get(config, "llm_max_retries", 2)),
        temperature=_optional_float(cfg_get(config, "llm_temperature", None)),
        max_tokens=int(cfg_get(config, "llm_max_tokens", 1024)),
        cache_dir=str(cache_dir) if cache_dir else None,
        response_format_json=bool(cfg_get(config, "llm_response_format_json", True)),
    )


def chat_json(messages: list[dict[str, str]], *, config: Any = None, purpose: str, cache_payload: Any) -> dict[str, Any]:
    llm_config = get_llm_config(config)
    cache_key = _cache_key(llm_config.model, purpose, cache_payload)
    cached = _read_cache(llm_config.cache_dir, cache_key)
    if cached is not None:
        _inc(f"{purpose}_cache_hits")
        return cached

    payload = {
        "model": llm_config.model,
        "messages": messages,
        "max_tokens": llm_config.max_tokens,
    }
    if llm_config.temperature is not None:
        payload["temperature"] = llm_config.temperature
    if llm_config.response_format_json:
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(max(1, llm_config.max_retries + 1)):
        try:
            content = _post_chat(llm_config, payload)
            data = _loads_json_object(content)
            _write_cache(llm_config.cache_dir, cache_key, data)
            _inc(f"{purpose}_requests")
            return data
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {400, 404, 422} and "response_format" in payload:
                payload = dict(payload)
                payload.pop("response_format", None)
            else:
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0, 0.25 * (attempt + 1)))

    _inc(f"{purpose}_errors")
    raise LLMUnavailable(f"LLM {purpose} request failed: {last_error}")


def warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message)


def get_llm_stats(reset: bool = True) -> dict[str, float]:
    with _STATS_LOCK:
        stats = {f"obliflow/llm_{key}": float(value) for key, value in _STATS.items()}
        if reset:
            _STATS.clear()
    return stats


def _post_chat(config: LLMConfig, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "api-key": config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMUnavailable(f"Unexpected chat response format: {data}") from exc


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data)}")
    return data


def _cache_key(model: str, purpose: str, payload: Any) -> str:
    raw = json.dumps({"model": model, "purpose": purpose, "payload": payload}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str | None, cache_key: str) -> dict[str, Any] | None:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{cache_key}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache(cache_dir: str | None, cache_key: str, data: dict[str, Any]) -> None:
    if not cache_dir:
        return
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    tmp_path = path / f"{cache_key}.{os.getpid()}.{threading.get_ident()}.tmp"
    final_path = path / f"{cache_key}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, final_path)


def _inc(key: str) -> None:
    with _STATS_LOCK:
        _STATS[key] = _STATS.get(key, 0) + 1


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    with _DOTENV_LOCK:
        if _DOTENV_LOADED:
            return
        candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
        for parent in Path.cwd().parents:
            candidates.append(parent / ".env")
        for path in candidates:
            if path.exists():
                _load_dotenv(path)
                _DOTENV_LOADED = True
                return
        _DOTENV_LOADED = True


def _load_dotenv(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
