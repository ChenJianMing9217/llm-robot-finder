"""選大腦：只換網址、金鑰、模型名稱，其他程式碼完全不動（OpenAI 相容介面）。

  LLM_PROVIDER=ollama（預設）：本機 Ollama，模型 qwen3:4b
  LLM_PROVIDER=remote：任何 OpenAI 相容的服務，另外設定 REMOTE_BASE_URL、REMOTE_API_KEY、REMOTE_MODEL
                        （推理模型要用工具時，有的服務要關掉推理：REMOTE_EXTRA='{"reasoning_effort": "none"}'）
金鑰可以放在環境變數，或同一個資料夾的 .env（已被 .gitignore 擋住），程式不會印出。
"""
import json
import os
from pathlib import Path

from openai import OpenAI


def _load_env():
    env = Path(__file__).resolve().parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()


def to_history(msg):
    """把模型的回覆轉成最標準的格式，再放回對話紀錄（各家「OpenAI 相容」對多出來或缺少的欄位容忍度不同）。"""
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def get(provider=None):
    """回傳 (client, 模型名稱, 額外參數)。"""
    _load_env()
    provider = provider or os.environ.get("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        return OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"), "qwen3:4b", {"temperature": 0}
    if provider == "remote":
        extra = json.loads(os.environ.get("REMOTE_EXTRA", "{}"))
        return (OpenAI(base_url=os.environ["REMOTE_BASE_URL"], api_key=os.environ["REMOTE_API_KEY"]),
                os.environ["REMOTE_MODEL"], extra)
    raise ValueError(f"不認得的 LLM_PROVIDER：{provider}（可以用 ollama 或 remote）")
