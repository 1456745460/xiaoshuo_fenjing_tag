#!/usr/bin/env python3
"""小说节点快照：读取 txt，调用 OpenAI 兼容接口生成 Anima 提示词。"""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "novel_to_anima_system_prompt.txt"
DEFAULT_NOVEL_PATH = ROOT / "samples" / "snapshot_test_excerpt.txt"
OUTPUT_PATH = ROOT / "test_output_deepseek.md"
OUTPUT_DIR = ROOT / "outputs"
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
SSL_CONTEXT = ssl._create_unverified_context()
DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_NOVEL_MAX_CHARS = 20000


def api_base(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        raise ValueError("API 地址不能为空")
    for suffix in ("/chat/completions", "/completions", "/models", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
    return base


def models_urls(base: str) -> list[str]:
    return [f"{base}/models", f"{base}/v1/models"]


def chat_urls(base: str) -> list[str]:
    return [f"{base}/chat/completions", f"{base}/v1/chat/completions"]


def http_json(
    method: str,
    url: str,
    *,
    api_key: str,
    body: dict | None = None,
    timeout: int = 60,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 {url}\n{exc.reason}") from exc
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回不是 JSON: {raw[:400]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"接口返回格式不对: {raw[:400]}")
    return parsed


def list_models(api_url: str, api_key: str) -> list[str]:
    key = api_key.strip()
    if not key:
        raise ValueError("API Key 不能为空")
    base = api_base(api_url)
    errors: list[str] = []
    for url in models_urls(base):
        try:
            payload = http_json("GET", url, api_key=key, timeout=30)
            data = payload.get("data") or []
            ids = []
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
                elif isinstance(item, str):
                    ids.append(item)
            if ids:
                return ids
            errors.append(f"{url} 没有返回模型列表")
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("获取模型失败：\n" + "\n".join(errors))


def resolve_novel_path(raw: str | None) -> Path:
    path = Path(raw).expanduser() if raw else DEFAULT_NOVEL_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到小说文件: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"路径不是文件: {path}")
    return path


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding if last_error else "utf-8",
        raw,
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"无法解码小说文件: {path}",
    )


def clip_novel(text: str, max_chars: int = DEFAULT_NOVEL_MAX_CHARS) -> tuple[str, bool]:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped, False
    chunk = stripped[:max_chars]
    cut = max(chunk.rfind("。"), chunk.rfind("！"), chunk.rfind("？"), chunk.rfind("\n"))
    if cut >= max_chars // 2:
        chunk = chunk[: cut + 1]
    return chunk.strip(), True


def build_user_prompt(novel: str, novel_name: str) -> str:
    return (
        "请根据下面这篇 txt 小说正文，严格执行系统提示词。\n"
        "本次输出 10 到 12 个关键节点。状态表写要点即可，"
        "但每个节点的中文提示词和英文提示词必须完整、可直接用于 Anima 文生图。\n"
        "必须先给角色一致性档案，再给节点。不要寒暄，不要解释用法。\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def extract_content(api_result: dict) -> str:
    choices = api_result.get("choices") or []
    if not choices:
        raise RuntimeError(f"接口返回没有 choices: {api_result}")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        reasoning = (message.get("reasoning_content") or "").strip()
        finish = choices[0].get("finish_reason")
        raise RuntimeError(
            "接口返回空 content。"
            f" finish_reason={finish}, reasoning_len={len(reasoning)}"
        )
    return content


def evaluate(content: str) -> list[str]:
    has_bible = "角色一致性档案" in content or "锁定英文" in content
    node_hits = re.findall(r"(?:^|\n)#*\s*节点\s*0?\d+", content)
    cn_hits = re.findall(r"中文提示词", content)
    en_hits = re.findall(r"英文提示词", content)
    tag_hits = re.findall(r"\b(?:1girl|1boy|natural_skin|full_body|upper_body)\b", content)
    quality_hits = re.findall(
        r"\b(?:masterpiece|best_quality|highres|absurdres)\b", content, flags=re.I
    )
    lines = [
        f"角色档案: {'有' if has_bible else '缺失'}",
        f"节点标题数: {len(node_hits)}",
        f"中文提示词块: {len(cn_hits)}",
        f"英文提示词块: {len(en_hits)}",
        f"关键 Anima tag 命中: {len(tag_hits)}",
        f"禁用质量 tag: {quality_hits or '无'}",
    ]
    ok = (
        has_bible
        and 10 <= max(len(node_hits), len(cn_hits), len(en_hits)) <= 15
        and len(tag_hits) >= 8
        and not quality_hits
    )
    lines.append(f"结构校验: {'通过' if ok else '未完全通过，请看正文'}")
    return lines


def chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout: int = 300,
) -> dict:
    key = api_key.strip()
    if not key:
        raise ValueError("API Key 不能为空")
    if not model.strip():
        raise ValueError("模型不能为空")
    base = api_base(api_url)
    payload = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": int(max_tokens),
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    errors: list[str] = []
    for url in chat_urls(base):
        current = dict(payload)
        for _ in range(2):
            try:
                return http_json(
                    "POST",
                    url,
                    api_key=key,
                    body=current,
                    timeout=timeout,
                )
            except RuntimeError as exc:
                message = str(exc)
                errors.append(message)
                if "thinking" in current and (
                    "thinking" in message.lower() or "HTTP 400" in message
                ):
                    current = {k: v for k, v in current.items() if k != "thinking"}
                    continue
                break
    raise RuntimeError("生成失败：\n" + "\n".join(errors))


def write_output(
    *,
    content: str,
    novel_path: Path,
    model: str,
    usage: dict,
    report: list[str],
    output_path: Path | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_path or (OUTPUT_DIR / f"{novel_path.stem}_{stamp}.md")
    header = [
        "# 小说节点快照",
        "",
        f"- 模型: `{model}`",
        f"- 小说: `{novel_path}`",
        f"- prompt_tokens: {usage.get('prompt_tokens', '未知')}",
        f"- completion_tokens: {usage.get('completion_tokens', '未知')}",
        f"- total_tokens: {usage.get('total_tokens', '未知')}",
        "",
        "## 结构校验",
        *[f"- {line}" for line in report],
        "",
        "## 模型正文",
        "",
        content.strip(),
        "",
    ]
    text = "\n".join(header)
    target.write_text(text, encoding="utf-8")
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    return target


def generate_snapshot(
    *,
    novel_path: Path,
    api_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    novel_max_chars: int = DEFAULT_NOVEL_MAX_CHARS,
) -> dict:
    system_prompt = load_text(PROMPT_PATH)
    novel, truncated = clip_novel(load_text(novel_path), novel_max_chars)
    if not novel:
        raise ValueError(f"小说文件是空的: {novel_path}")
    user_prompt = build_user_prompt(novel, novel_path.name)
    result = chat_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )
    content = extract_content(result)
    usage = result.get("usage") or {}
    report = evaluate(content)
    output_path = write_output(
        content=content,
        novel_path=novel_path,
        model=model,
        usage=usage,
        report=report,
    )
    return {
        "content": content,
        "usage": usage,
        "report": report,
        "output_path": output_path,
        "truncated": truncated,
        "novel_chars": len(novel),
        "system_chars": len(system_prompt),
    }


def open_path(path: Path) -> None:
    target = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
    elif sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", target])


def open_dir(path: Path) -> None:
    folder = path if path.is_dir() else path.parent
    open_path(folder)
