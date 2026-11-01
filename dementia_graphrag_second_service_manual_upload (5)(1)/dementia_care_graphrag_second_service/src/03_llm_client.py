"""
Step 03: DeepSeek LLM client.

Stable version:
- No persistent HTTP client/session is stored.
- Each request creates a fresh requests.Session and closes it after use.
- Safe to be used inside cached mapper/runtime objects.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_json_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        load_dotenv(PROJECT_ROOT / ".env")

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = int(timeout or os.getenv("DEEPSEEK_TIMEOUT", "60"))
        self.max_retries = int(max_retries or os.getenv("DEEPSEEK_MAX_RETRIES", "3"))

        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is empty. Please set it in .env.")

    def _make_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        if messages is not None:
            return messages

        result: List[Dict[str, str]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        if user_prompt:
            result.append({"role": "user", "content": user_prompt})

        if not result:
            raise ValueError("Either messages or user_prompt must be provided.")

        return result

    def _request_chat(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
            "User-Agent": "curl/8.0",
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            session = requests.Session()
            session.trust_env = False

            try:
                resp = session.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )

                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"DeepSeek API error {resp.status_code}: {resp.text[:1000]}"
                    )

                return resp.json()

            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break

                time.sleep(min(2 * attempt, 6))

            finally:
                try:
                    session.close()
                except Exception:
                    pass

        raise RuntimeError(f"DeepSeek request failed after retries: {repr(last_error)}")

    def chat_text(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._make_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = self._request_chat(payload)
        return data["choices"][0]["message"]["content"].strip()

    def chat_json(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._make_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        data = self._request_chat(payload)
        content = data["choices"][0]["message"]["content"]
        return clean_json_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["text", "json"], default="json")
    args = parser.parse_args()

    client = DeepSeekClient()

    if args.test == "text":
        out = client.chat_text(
            system_prompt="你是一个测试助手。",
            user_prompt="请回复：你好",
            max_tokens=100,
        )
        print(out)

    else:
        raw = client.chat_json(
            system_prompt="你只输出 JSON。",
            user_prompt='请输出 {"task":"test","ok":true,"message":"你好"}',
            max_tokens=200,
        )
        print(raw)
        print(json.loads(raw))

    print("[PASS] 03_llm_client.py test finished.")


if __name__ == "__main__":
    main()
