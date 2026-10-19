import base64
import io
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Tuple

import requests


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def _pil_to_bytes_and_mime(img: Any) -> Tuple[bytes, str]:
    """
    Convert a PIL image object to bytes + mime.
    We keep this helper separate so api providers can accept either
    file paths or PIL images (datasets sometimes load images eagerly).
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise TypeError(
            "image_file is not a path string, and Pillow (PIL) is not available "
            "to serialize the in-memory image."
        ) from e

    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(img)!r}")

    fmt = (getattr(img, "format", None) or "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    mime_by_fmt = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }
    mime = mime_by_fmt.get(fmt, "image/jpeg")

    buf = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs = {"quality": 95, "optimize": True}
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue(), mime


def _image_to_bytes_and_mime(image_file: Any) -> Tuple[bytes, str]:
    """
    Accept either:
      - str/pathlike -> read file
      - PIL.Image.Image -> serialize to bytes
    """
    if isinstance(image_file, (str, bytes, os.PathLike)):
        path = os.fspath(image_file)
        mime = _guess_mime(path)
        with open(path, "rb") as f:
            return f.read(), mime
    return _pil_to_bytes_and_mime(image_file)


def _image_to_data_url(image_file: Any) -> str:
    data, mime = _image_to_bytes_and_mime(image_file)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@dataclass
class OpenAIVisionProvider:
    """
    model_path format: openai/<model>

    Auth:
      - export OPENAI_API_KEY=...
    Optional:
      - export OPENAI_API_BASE=https://api.openai.com/v1
    """

    model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_base_env: str = "OPENAI_API_BASE"

    def __call__(self, image_file: Any, query: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it in your environment.")
        api_base = os.getenv(self.api_base_env, "https://api.openai.com/v1").rstrip("/")

        # Official: Responses API with input_image using a data URL.
        # Docs: https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": query},
                        {"type": "input_image", "image_url": _image_to_data_url(image_file)},
                    ],
                }
            ],
        }
        r = requests.post(
            f"{api_base}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        # Prefer output_text if present; otherwise best-effort extract.
        if isinstance(data, dict) and "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        # Fallback: try common shapes.
        try:
            out = data.get("output", [])
            for item in out:
                if item.get("type") in ("message", "output_text"):
                    if "content" in item:
                        # message content array
                        parts = item["content"]
                        if isinstance(parts, list):
                            texts = []
                            for p in parts:
                                if isinstance(p, dict) and p.get("type") in ("output_text", "text"):
                                    t = p.get("text") or p.get("content")
                                    if t:
                                        texts.append(t)
                            if texts:
                                return "\n".join(texts)
        except Exception:
            pass
        return str(data)


@dataclass
class GeminiVisionProvider:
    """
    model_path format: gemini/<model>

    Auth:
      - export GEMINI_API_KEY=...
    Optional:
      - export GEMINI_API_BASE=https://generativelanguage.googleapis.com

    Endpoint:
      POST /v1beta/models/{model}:generateContent
    """

    model: str
    api_key_env: str = "GEMINI_API_KEY"
    api_base_env: str = "GEMINI_API_BASE"

    def __call__(self, image_file: Any, query: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it in your environment.")
        api_base = os.getenv(self.api_base_env, "https://generativelanguage.googleapis.com").rstrip("/")

        data, mime = _image_to_bytes_and_mime(image_file)
        b64 = base64.b64encode(data).decode("utf-8")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": mime, "data": b64}},
                        {"text": query},
                    ],
                }
            ]
        }
        r = requests.post(
            f"{api_base}/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
                if texts:
                    return "\n".join(texts)
        except Exception:
            pass
        return str(data)


@dataclass
class ZhipuVisionProvider:
    """
    model_path format: zhipu/<model>  (e.g., zhipu/glm-5v-turbo)

    Auth:
      - export ZHIPU_API_KEY=...
    Optional:
      - export ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4

    Endpoint:
      POST /chat/completions
    """

    model: str
    api_key_env: str = "ZHIPU_API_KEY"
    api_base_env: str = "ZHIPU_API_BASE"

    def __call__(self, image_file: Any, query: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it in your environment.")
        api_base = os.getenv(self.api_base_env, "https://open.bigmodel.cn/api/paas/v4").rstrip("/")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_file)}},
                        {"type": "text", "text": query},
                    ],
                }
            ],
        }
        r = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        try:
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        except Exception:
            pass
        return str(data)


@dataclass
class ParityVisionProvider:
    """
    OpenAI-compatible /chat/completions vision provider.

    Use this for any gateway that proxies multiple providers behind a single
    OpenAI-compatible chat endpoint. Handles `gpt-*` and `gemini-*` model ids
    alike — both via `/chat/completions` with `image_url` data URL content.

    model_path format: uniapi/<model>  (or openai/, gemini/, ...)

    Auth:
      - export UNIAPI_API_KEY=...   (falls back to OPENAI_API_KEY)
      - export UNIAPI_API_BASE=...  (falls back to OPENAI_API_BASE)
    """

    model: str
    api_key_env: str = "UNIAPI_API_KEY"
    api_base_env: str = "UNIAPI_API_BASE"

    def __call__(self, image_file: Any, query: str) -> str:
        api_key = os.getenv(self.api_key_env) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Missing {self.api_key_env} (and OPENAI_API_KEY fallback). "
                "Set one of them in your environment."
            )
        api_base = (os.getenv(self.api_base_env) or os.getenv("OPENAI_API_BASE")
                    or "https://api.openai.com/v1").rstrip("/")

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url",
                     "image_url": {"url": _image_to_data_url(image_file)}},
                ],
            }],
        }
        r = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        try:
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        p.get("text") for p in content
                        if isinstance(p, dict) and p.get("type") in ("text", "output_text")
                    ]
                    if any(parts):
                        return "\n".join(p for p in parts if p)
        except Exception:
            pass
        return str(data)


@dataclass
class MiniMaxVisionProvider:
    """
    model_path format: minimax/<model>

    MiniMax's official "Text Chat V2" endpoint is text-only.
    This class exists as a placeholder so you can later wire in MiniMax's
    multimodal/vision API once you decide which official product to use.

    Auth placeholder:
      - export MINIMAX_API_KEY=...
    Optional placeholder:
      - export MINIMAX_API_BASE=https://api.minimax.io
      - export MINIMAX_API_PATH=/v1/text/chatcompletion_v2
    """

    model: str
    api_key_env: str = "MINIMAX_API_KEY"
    api_base_env: str = "MINIMAX_API_BASE"
    api_path_env: str = "MINIMAX_API_PATH"

    def __call__(self, image_file: str, query: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it in your environment.")

        api_base = os.getenv(self.api_base_env, "https://api.minimax.io").rstrip("/")
        api_path = os.getenv(self.api_path_env, "/v1/text/chatcompletion_v2")

        raise NotImplementedError(
            "MiniMax vision is not wired yet. The official Text Chat V2 endpoint is text-only. "
            "Once you have MiniMax's official multimodal endpoint + request schema, implement it here "
            f"(base={api_base!r}, path={api_path!r})."
        )

