#!/usr/bin/env python3
"""小说节点快照：读取 txt，调用 Chat Completions 或 Responses 接口生成提示词。"""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_STYLE_DANBOORU = "danbooru"
PROMPT_STYLE_NATURAL = "natural"
PROMPT_STYLE_KREA2 = "krea2"
DEFAULT_PROMPT_STYLE = PROMPT_STYLE_DANBOORU
PROMPT_FILES = {
    PROMPT_STYLE_DANBOORU: ROOT / "novel_to_anima_system_prompt.txt",
    PROMPT_STYLE_NATURAL: ROOT / "novel_to_nl_system_prompt.txt",
    PROMPT_STYLE_KREA2: ROOT / "novel_to_krea2_system_prompt.txt",
}
PROMPT_STYLE_LABELS = {
    PROMPT_STYLE_DANBOORU: "Danbooru",
    PROMPT_STYLE_NATURAL: "自然语言",
    PROMPT_STYLE_KREA2: "Krea2",
}
PROMPT_PATH = PROMPT_FILES[PROMPT_STYLE_DANBOORU]
NL_CHARACTER_PROMPT_PATH = ROOT / "novel_to_nl_character_system_prompt.txt"
NL_SUMMARY_PROMPT_PATH = ROOT / "novel_to_nl_summary_system_prompt.txt"
DEFAULT_NOVEL_PATH = ROOT / "samples" / "snapshot_test_excerpt.txt"
OUTPUT_PATH = ROOT / "test_output_deepseek.md"
OUTPUT_DIR = ROOT / "outputs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediates"
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
SSL_CONTEXT = ssl._create_unverified_context()
DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_NOVEL_MAX_CHARS = 20000
MIN_NODE_COUNT = 1
MAX_NODE_COUNT = 100
DEFAULT_NODE_COUNT = 12
NODE_COUNT_PLACEHOLDER = "{{NODE_COUNT}}"
API_BACKEND_AUTO = "auto"
API_BACKEND_CHAT = "chat_completions"
API_BACKEND_RESPONSES = "responses"
DEFAULT_API_BACKEND = API_BACKEND_AUTO
DEFAULT_USER_AGENT = "xiaoshuo-fenjing-tag/1.0"
API_BACKEND_LABELS = {
    API_BACKEND_AUTO: "自动",
    API_BACKEND_CHAT: "Chat Completions",
    API_BACKEND_RESPONSES: "Responses",
}
API_BACKEND_CHOICES = [
    (API_BACKEND_AUTO, "自动（Grok 用 Responses，DeepSeek 用 Chat）"),
    (API_BACKEND_CHAT, "Chat Completions"),
    (API_BACKEND_RESPONSES, "Responses"),
]
_PROMPT_STYLE_ALIASES = {
    "danbooru": PROMPT_STYLE_DANBOORU,
    "tag": PROMPT_STYLE_DANBOORU,
    "tags": PROMPT_STYLE_DANBOORU,
    "anima": PROMPT_STYLE_DANBOORU,
    "natural": PROMPT_STYLE_NATURAL,
    "nl": PROMPT_STYLE_NATURAL,
    "natural_language": PROMPT_STYLE_NATURAL,
    "自然语言": PROMPT_STYLE_NATURAL,
    "krea2": PROMPT_STYLE_KREA2,
    "krea": PROMPT_STYLE_KREA2,
    "krea_2": PROMPT_STYLE_KREA2,
    "krea-2": PROMPT_STYLE_KREA2,
    "k2": PROMPT_STYLE_KREA2,
}


def api_base(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        raise ValueError("API 地址不能为空")
    for suffix in ("/chat/completions", "/completions", "/responses", "/models"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return base


def endpoint_urls(base: str, path: str) -> list[str]:
    urls = [f"{base}/{path}"]
    if not base.endswith("/v1"):
        urls.append(f"{base}/v1/{path}")
    return urls


def models_urls(base: str) -> list[str]:
    return endpoint_urls(base, "models")


def chat_urls(base: str) -> list[str]:
    return endpoint_urls(base, "chat/completions")


def responses_urls(base: str) -> list[str]:
    return endpoint_urls(base, "responses")


def infer_api_backend(model: str) -> str:
    name = (model or "").strip().lower()
    if name.startswith("grok") or name.startswith("xai/") or "/grok" in name:
        return API_BACKEND_RESPONSES
    return API_BACKEND_CHAT


def normalize_api_backend(raw: str | None, model: str = "") -> str:
    text = (raw or DEFAULT_API_BACKEND).strip().lower().replace("-", "_")
    aliases = {
        "": API_BACKEND_AUTO,
        "auto": API_BACKEND_AUTO,
        "automatic": API_BACKEND_AUTO,
        "自动": API_BACKEND_AUTO,
        "chat": API_BACKEND_CHAT,
        "chat_completion": API_BACKEND_CHAT,
        "chat_completions": API_BACKEND_CHAT,
        "openai": API_BACKEND_CHAT,
        "completions": API_BACKEND_CHAT,
        "deepseek": API_BACKEND_CHAT,
        "responses": API_BACKEND_RESPONSES,
        "response": API_BACKEND_RESPONSES,
        "grok": API_BACKEND_RESPONSES,
    }
    if text not in aliases:
        raise ValueError(
            f"不支持的 API 协议: {raw}（可选 auto / chat_completions / responses）"
        )
    backend = aliases[text]
    if backend == API_BACKEND_AUTO:
        return infer_api_backend(model)
    return backend


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
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
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


def normalize_prompt_style(raw: str | None) -> str:
    text = (raw or DEFAULT_PROMPT_STYLE).strip().lower()
    style = _PROMPT_STYLE_ALIASES.get(text)
    if style is None:
        raise ValueError(f"不支持的提示词类型: {raw}（可选 danbooru / natural / krea2）")
    return style


def prompt_path_for(style: str) -> Path:
    path = PROMPT_FILES[normalize_prompt_style(style)]
    if not path.exists():
        raise FileNotFoundError(f"找不到系统提示词: {path}")
    return path


def normalize_node_count(raw: object) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"分镜数必须是 {MIN_NODE_COUNT} 到 {MAX_NODE_COUNT} 的整数"
        ) from exc
    if value < MIN_NODE_COUNT or value > MAX_NODE_COUNT:
        raise ValueError(
            f"分镜数必须是 {MIN_NODE_COUNT} 到 {MAX_NODE_COUNT} 的整数"
        )
    return value


def render_system_prompt(text: str, node_count: int | None = None) -> str:
    if NODE_COUNT_PLACEHOLDER not in text:
        return text
    if node_count is None:
        raise ValueError(f"系统提示词含 {NODE_COUNT_PLACEHOLDER}，但未提供分镜数")
    count = normalize_node_count(node_count)
    return text.replace(NODE_COUNT_PLACEHOLDER, str(count))


def emit_progress(on_progress: Callable[[str], None] | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


def normalize_usage(usage: dict | None) -> dict:
    data = dict(usage or {})
    if "prompt_tokens" not in data and isinstance(data.get("input_tokens"), int):
        data["prompt_tokens"] = data["input_tokens"]
    if "completion_tokens" not in data and isinstance(data.get("output_tokens"), int):
        data["completion_tokens"] = data["output_tokens"]
    if "total_tokens" not in data:
        prompt = data.get("prompt_tokens")
        completion = data.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            data["total_tokens"] = prompt + completion
    return data


def merge_usage(*usages: dict) -> dict:
    merged: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total = 0
        found = False
        for usage in usages:
            value = normalize_usage(usage).get(key)
            if isinstance(value, int):
                total += value
                found = True
        if found:
            merged[key] = total
    return merged


def clip_novel(text: str, max_chars: int = DEFAULT_NOVEL_MAX_CHARS) -> tuple[str, bool]:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped, False
    chunk = stripped[:max_chars]
    cut = max(chunk.rfind("。"), chunk.rfind("！"), chunk.rfind("？"), chunk.rfind("\n"))
    if cut >= max_chars // 2:
        chunk = chunk[: cut + 1]
    return chunk.strip(), True


def build_user_prompt(
    novel: str,
    novel_name: str,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
) -> str:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    if style == PROMPT_STYLE_KREA2:
        prompt_rule = (
            "每个节点的英文提示词必须是可直接贴进 Krea 2 的导演简报，80 到 160 词。"
            "必须先写全书风格锁定句（2D anime still frame, modern TV anime cel, visible linework, cel-shaded 等），再写机位/人数，再按人写锁定外貌+此刻服装动作，再写场景、主光、赛璐璐阴影。"
            "禁止 photoreal / live-action / cinematic still / 35mm / film grain。禁止 Danbooru tag、snake_case、BREAK、1girl、solo。中文提示词信息对齐，供阅读；出图只贴英文。"
            "人数用肯定句：frame holds only N people（N≤3）。不要堆 no crowd / no bystanders 否定清单。"
            "每个分镜最多 3 个可识别人物；正文超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
            "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把人数加一。"
            "多人必须叙事构图：过肩/侧面/面对面，写清谁看谁，禁止并排看镜头，禁止 looking at the camera。"
            "表情写五官动作；女性爱按强度用愉悦/高潮；男色欲用邪笑/得意/坏笑，不要乱加。"
        )
    elif style == PROMPT_STYLE_NATURAL:
        prompt_rule = (
            "中文提示词必须是完整画面描述；英文必须是 Anima 混合写法："
            "标签块（masterpiece, best quality, score_7 + 1girl/1boy + 按人外观服装）空一行后再写短句。"
            "禁止英文散文、snake_case、BREAK、(word:1.2)、looking_at_viewer。"
            "多人同框必须一人一句，主语带外貌锚点，独有特征禁止写成全画面清单。"
            "段首先锁人数：画面里只有 N 个人（N≤3）。每个分镜最多 3 个可识别人物；正文超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
            "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把两人写成三人、三人写成四人。"
            "多人必须叙事构图：面对面或侧面相对、写清谁看谁，禁止并排看镜头合影，禁止直视镜头。"
            "表情写五官动作，禁止只写微笑/脸红；女性爱按强度用愉悦表情/高潮脸/阿嘿颜等，男色欲用邪笑/得意/坏笑，不要乱加。"
        )
    else:
        prompt_rule = (
            "每个节点的中文提示词和英文提示词必须完整、可直接用于 Anima 文生图。"
            "英文必须是 Danbooru / snake_case tag，逗号分隔，禁止写成自然语言句子。"
            "多人同框必须用 BREAK 分角色块，独有特征禁止写进全局段，并写清左右站位和互斥项。"
            "人数 tag 必须精确，且每镜最多 3 人；正文超过 3 人必须拆镜或只留核心 1～3 人，禁止 4girls / multiple_girls 群像。"
            "禁止 reflection / married / wife / husband / crowd 把人数加一；公共场景补 empty, no_crowd。"
            "多人必须 from_side / facing_each_other / looking_at_another，禁止 looking_at_viewer 合影。"
            "女快感按强度用 pleasure_face / ahegao 等，男色欲用 evil_smile / smug / smirk，禁止只写 smile。"
        )
    return (
        "请根据下面这篇 txt 小说正文，严格执行系统提示词。\n"
        f"本次必须输出正好 {count} 个关键节点，不多不少。状态表写要点即可，"
        f"但{prompt_rule}\n"
        "必须先给角色一致性档案，再给节点。不要寒暄，不要解释用法。\n"
        "防串台：禁止把发色、眼镜、服装混成一袋；一人戴眼镜则另一人必须明确不戴。\n"
        "防人数膨胀：每个分镜最多 3 个可识别人物，禁止 4 人同框；在场几人就只写几人；玻璃/镜子只写光斑不写人物倒影；锁定外貌禁止 married/妻子/丈夫；走廊办公室默认空场。\n"
        "防合影：多人禁止看镜头，必须对视或看对方身体，机位用侧面/过肩/面对面，不要正面并排。\n"
        "表情：写五官动作；女性爱按强度选愉悦/高潮/阿嘿颜；男色欲用邪笑/得意/坏笑，禁止一律微笑，禁止乱加。\n"
        "角色必须一眼能分清：正文没写死时，不同人要错开发色、长短发、身材、胡子/眼镜/痣；禁止全员同一张脸。\n"
        "服装必须写颜色、花纹、面料纹理、剪裁，禁止只写衬衫/裙子/white_shirt。\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def build_nl_character_user_prompt(novel: str, novel_name: str) -> str:
    return (
        "请根据下面这篇 txt 小说正文，只产出「角色一致性档案」。\n"
        "不要写节点，不要写中英文提示词，不要写剧情摘要。不要寒暄。\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def build_nl_summary_user_prompt(novel: str, novel_name: str) -> str:
    return (
        "请根据下面这篇 txt 小说正文，写出保留完整故事结构的详细概括。\n"
        "不要过于精简。严格保留整体故事结构、不同爱情姿势、不同爱情经过。\n"
        "每一场情爱单独写，每换一种姿势就分条写，禁止合并成「两人做爱」。\n"
        "不要写角色外貌档案，不要写分镜提示词。不要寒暄。\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def build_nl_tag_user_prompt(
    character_bible: str,
    summary: str,
    novel_name: str,
    node_count: int,
) -> str:
    count = normalize_node_count(node_count)
    prompt_rule = (
        "中文提示词必须是完整画面描述，一人一句，主语带外貌锚点。"
        "英文提示词必须是 Anima 混合写法：先写小写空格分词的标签块"
        "（masterpiece, best quality, score_7，NSFW 加 explicit，再写 1girl/1boy 和按人分行的外观服装），"
        "空一行后写 4 到 6 句短英文（构图、左、右、光影、归属）。"
        "禁止把英文写成一篇嵌套从句散文。禁止 snake_case、BREAK、(word:1.2)、looking_at_viewer。"
        "锁定外貌要素写进英文标签段；短句只回锚 2～4 个辨识点，不要把锁定句整段嵌进从句。"
        "段首先锁人数：画面里只有 N 个人（N≤3）。每个分镜最多 3 个可识别人物；"
        "概括超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
        "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把两人写成三人、三人写成四人。"
        "多人必须叙事构图：面对面或侧面相对、写清谁看谁，禁止并排看镜头合影，禁止直视镜头。"
        "表情写五官动作，禁止只写微笑/脸红；女性爱按强度用愉悦表情/高潮脸/阿嘿颜等，"
        "男色欲用邪笑/得意/坏笑，不要乱加。"
    )
    return (
        "请根据下面已经完成的「角色一致性档案」和「小说内容概括」，严格执行系统提示词。\n"
        f"本次必须输出正好 {count} 个关键节点，不多不少。状态表写要点即可，"
        f"但{prompt_rule}\n"
        "不要再读原文，不要重写人物外貌。锁定外貌句必须原样粘贴。\n"
        "不要再输出角色一致性档案，不要再输出小说概括。直接从节点目录写起。\n"
        "概括里的不同爱情姿势、不同爱情经过必须尽量用不同节点覆盖，禁止合并省略。\n"
        "防串台：禁止把发色、眼镜、服装混成一袋；一人戴眼镜则另一人必须明确不戴。\n"
        "防人数膨胀：每个分镜最多 3 个可识别人物，禁止 4 人同框；在场几人就只写几人；"
        "玻璃/镜子只写光斑不写人物倒影；锁定外貌禁止 married/妻子/丈夫；走廊办公室默认空场。\n"
        "防合影：多人禁止看镜头，必须对视或看对方身体，机位用侧面/过肩/面对面，不要正面并排。\n"
        "表情：写五官动作；女性爱按强度选愉悦/高潮/阿嘿颜；男色欲用邪笑/得意/坏笑，禁止一律微笑，禁止乱加。\n"
        "服装必须写颜色、花纹、面料纹理、剪裁，禁止只写衬衫/裙子/white_shirt。\n"
        "以下材料可能已经过人工修改，一律以本次给定文本为准，不要用旧版记忆。\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<character_bible>\n"
        f"{character_bible.strip()}\n"
        "</character_bible>\n\n"
        "<novel_summary>\n"
        f"{summary.strip()}\n"
        "</novel_summary>"
    )


def _text_from_parts(parts: object) -> str:
    chunks: list[str] = []
    if isinstance(parts, str):
        return parts.strip()
    if not isinstance(parts, list):
        return ""
    for part in parts:
        if isinstance(part, str) and part.strip():
            chunks.append(part.strip())
        elif isinstance(part, dict):
            kind = str(part.get("type") or "")
            if kind in ("output_text", "text") or not kind:
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()


def extract_content(api_result: dict) -> str:
    output_text = api_result.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in api_result.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if kind == "reasoning":
            continue
        if kind == "message":
            text = _text_from_parts(item.get("content"))
        elif kind in ("output_text", "text"):
            text = str(item.get("text") or "").strip()
        else:
            text = _text_from_parts(item.get("content"))
        if text:
            texts.append(text)
    if texts:
        return "\n\n".join(texts)

    choices = api_result.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if content:
            return content
        reasoning = (message.get("reasoning_content") or "").strip()
        finish = choices[0].get("finish_reason")
        raise RuntimeError(
            "接口返回空 content。"
            f" finish_reason={finish}, reasoning_len={len(reasoning)}"
        )

    status = api_result.get("status") or "未知"
    error = api_result.get("error")
    raise RuntimeError(
        f"接口返回没有正文。status={status}, error={error}"
    )


def evaluate(
    content: str,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
) -> list[str]:
    style = normalize_prompt_style(prompt_style)
    expected = normalize_node_count(node_count)
    has_bible = "角色一致性档案" in content or "锁定英文" in content
    node_hits = re.findall(r"(?:^|\n)#*\s*节点\s*0?\d+", content)
    cn_hits = re.findall(r"中文提示词", content)
    en_hits = re.findall(r"英文提示词", content)
    tag_hits = re.findall(r"\b(?:1girl|1boy|natural_skin|full_body|upper_body)\b", content)
    quality_hits = re.findall(
        r"\b(?:masterpiece|best_quality|best quality|highres|absurdres)\b",
        content,
        flags=re.I,
    )
    mixed_quality = re.findall(r"\b(?:masterpiece|best quality|score_7)\b", content, flags=re.I)
    found_count = max(len(node_hits), len(cn_hits), len(en_hits))
    lines = [
        f"提示词类型: {PROMPT_STYLE_LABELS[style]}",
        f"要求分镜数: {expected}",
        f"角色档案: {'有' if has_bible else '缺失'}",
        f"节点标题数: {len(node_hits)}",
        f"中文提示词块: {len(cn_hits)}",
        f"英文提示词块: {len(en_hits)}",
    ]
    if style == PROMPT_STYLE_NATURAL:
        count_hits = re.findall(r"\b(?:1girl|1boy|2girls|3girls)\b", content)
        snake_hits = re.findall(
            r"\b(?:black_hair|long_hair|looking_at_viewer|natural_skin|full_body)\b",
            content,
        )
        lines.append(f"Anima 混合质量锚点: {len(mixed_quality)}")
        lines.append(f"Anima 人数 tag: {len(count_hits)}")
        lines.append(f"snake_case 残留: {len(snake_hits)}")
        ok = has_bible and found_count == expected
        if len(mixed_quality) < expected:
            lines.append("警告: 英文提示词缺少 masterpiece / best quality / score_7 前缀")
            ok = False
        if len(count_hits) < expected:
            lines.append("警告: 英文提示词缺少 1girl/1boy 人数锚点，Anima 混合写法会变弱")
            ok = False
        if snake_hits:
            lines.append("警告: 英文仍有 snake_case，Anima 混合写法应改用空格分词")
            ok = False
        if "BREAK" in content:
            lines.append("警告: 英文出现 BREAK，第三步混合写法不要用纯 Danbooru 分块")
            ok = False
    elif style == PROMPT_STYLE_KREA2:
        lines.append(f"禁用质量套话: {quality_hits or '无'}")
        lines.append(f"Danbooru tag 残留: {len(tag_hits)}")
        ok = has_bible and found_count == expected and not quality_hits
        if len(tag_hits) >= 8:
            lines.append("警告: 正文里仍出现较多 Danbooru tag，请核对是否按 Krea2 导演简报输出")
            ok = False
        if "全书视觉锁定" not in content:
            lines.append("警告: 未找到全书视觉锁定，Krea2 容易漂风格")
            ok = False
    else:
        lines.append(f"禁用质量套话: {quality_hits or '无'}")
        lines.append(f"关键 Anima tag 命中: {len(tag_hits)}")
        ok = has_bible and found_count == expected and not quality_hits and len(tag_hits) >= 8
    lines.append(f"结构校验: {'通过' if ok else '未完全通过，请看正文'}")
    return lines


def evaluate_nl_pipeline(
    character_bible: str,
    summary: str,
    nodes: str,
    node_count: int,
) -> list[str]:
    has_bible = "角色一致性档案" in character_bible or "锁定中文外貌" in character_bible
    has_summary = "小说内容概括" in summary or "情爱场次" in summary or "故事骨架" in summary
    lines = [
        "流水线: 自然语言三步独立对话",
        f"第一步人物一致性: {'有' if has_bible else '缺失'}",
        f"第二步小说概括: {'有' if has_summary else '缺失'}",
    ]
    assembled = f"{character_bible.strip()}\n\n{nodes.strip()}"
    lines.extend(evaluate(assembled, PROMPT_STYLE_NATURAL, node_count))
    if not has_summary:
        lines.append("警告: 第二步概括缺少标题或情爱场次，后续分镜可能丢姿势")
        if lines and lines[-2].startswith("结构校验:"):
            lines[-2] = "结构校验: 未完全通过，请看正文"
        elif lines[-1].startswith("结构校验:"):
            lines[-1] = "结构校验: 未完全通过，请看正文"
    return lines


def _is_protocol_error(message: str) -> bool:
    lower = message.lower()
    return (
        "protocol_not_supported" in lower
        or "不支持 chat completions" in message
        or "不支持 chat completion" in message
        or "does not support chat completions" in lower
        or "unsupported protocol" in lower
        or "unknown endpoint" in lower
    )


def _post_payloads(
    *,
    urls: list[str],
    payloads: list[dict],
    api_key: str,
    timeout: int,
    errors: list[str],
) -> dict:
    for url in urls:
        for payload in payloads:
            try:
                return http_json(
                    "POST",
                    url,
                    api_key=api_key,
                    body=payload,
                    timeout=timeout,
                )
            except RuntimeError as exc:
                errors.append(str(exc))
    raise RuntimeError("生成失败：\n" + "\n".join(errors))


def _chat_payloads(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> list[dict]:
    base = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    return [
        {**base, "thinking": {"type": "disabled"}},
        base,
    ]


def _responses_payloads(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> list[dict]:
    simple = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": int(max_tokens),
    }
    return [
        {**simple, "temperature": 0.3, "stream": False},
        simple,
    ]


def chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout: int | None = None,
    api_backend: str | None = None,
) -> dict:
    key = api_key.strip()
    if not key:
        raise ValueError("API Key 不能为空")
    model_name = model.strip()
    if not model_name:
        raise ValueError("模型不能为空")
    backend = normalize_api_backend(api_backend, model_name)
    wait = timeout if timeout is not None else (600 if backend == API_BACKEND_RESPONSES else 300)
    base = api_base(api_url)
    errors: list[str] = []
    attempts = [backend]
    if (api_backend or DEFAULT_API_BACKEND) in ("", API_BACKEND_AUTO, "自动", "auto"):
        fallback = (
            API_BACKEND_CHAT
            if backend == API_BACKEND_RESPONSES
            else API_BACKEND_RESPONSES
        )
        if fallback not in attempts:
            attempts.append(fallback)

    for current_backend in attempts:
        urls = responses_urls(base) if current_backend == API_BACKEND_RESPONSES else chat_urls(base)
        payloads = (
            _responses_payloads(model_name, system_prompt, user_prompt, max_tokens)
            if current_backend == API_BACKEND_RESPONSES
            else _chat_payloads(model_name, system_prompt, user_prompt, max_tokens)
        )
        try:
            return _post_payloads(
                urls=urls,
                payloads=payloads,
                api_key=key,
                timeout=wait,
                errors=errors,
            )
        except RuntimeError as exc:
            message = str(exc)
            if current_backend == attempts[-1] or (
                current_backend == backend and not _is_protocol_error(message)
            ):
                raise
            continue
    raise RuntimeError("生成失败：\n" + "\n".join(errors))


def write_output(
    *,
    content: str,
    novel_path: Path,
    model: str,
    usage: dict,
    report: list[str],
    output_path: Path | None = None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
    extra_header: list[str] | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_path or (OUTPUT_DIR / f"{novel_path.stem}_{stamp}.md")
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    header = [
        "# 小说节点快照",
        "",
        f"- 模型: `{model}`",
        f"- 小说: `{novel_path}`",
        f"- 提示词类型: `{PROMPT_STYLE_LABELS[style]}` ({style})",
        f"- 分镜数: `{count}`",
        *(extra_header or []),
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


def nl_draft_paths(novel_path: Path) -> tuple[Path, Path]:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    stem = novel_path.stem
    return (
        INTERMEDIATE_DIR / f"{stem}_character.md",
        INTERMEDIATE_DIR / f"{stem}_summary.md",
    )


def write_nl_character_draft(novel_path: Path, character_bible: str) -> Path:
    character_path, _ = nl_draft_paths(novel_path)
    character_path.write_text(character_bible.strip() + "\n", encoding="utf-8")
    return character_path


def write_nl_summary_draft(novel_path: Path, summary: str) -> Path:
    _, summary_path = nl_draft_paths(novel_path)
    summary_path.write_text(summary.strip() + "\n", encoding="utf-8")
    return summary_path


def write_nl_drafts(
    novel_path: Path,
    character_bible: str,
    summary: str,
) -> tuple[Path, Path]:
    return (
        write_nl_character_draft(novel_path, character_bible),
        write_nl_summary_draft(novel_path, summary),
    )


def load_nl_drafts(novel_path: Path) -> tuple[str, str]:
    character_path, summary_path = nl_draft_paths(novel_path)
    if not character_path.exists():
        raise FileNotFoundError(f"找不到人物一致性稿: {character_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到故事概括稿: {summary_path}")
    character_bible = load_text(character_path).strip()
    summary = load_text(summary_path).strip()
    if not character_bible:
        raise ValueError(f"人物一致性稿是空的: {character_path}")
    if not summary:
        raise ValueError(f"故事概括稿是空的: {summary_path}")
    return character_bible, summary


def run_chat_step(
    *,
    step_name: str,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    on_progress: Callable[[str], None] | None = None,
    api_backend: str | None = None,
) -> tuple[str, dict]:
    emit_progress(on_progress, step_name)
    result = chat_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        api_backend=api_backend,
    )
    content = extract_content(result)
    if not content.strip():
        raise RuntimeError(f"{step_name} 返回空内容")
    return content, normalize_usage(result.get("usage") or {})


def generate_nl_tag_from_materials(
    *,
    novel_path: Path,
    character_bible: str,
    summary: str,
    api_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    node_count: int,
    on_progress: Callable[[str], None] | None = None,
    pipeline_label: str = "自然语言第三步（按当前人物一致性与概括出 TAG）",
    prior_usages: list[dict] | None = None,
    truncated: bool = False,
    novel_chars: int = 0,
    extra_system_chars: int = 0,
    api_backend: str | None = None,
) -> dict:
    bible = character_bible.strip()
    plot = summary.strip()
    if not bible:
        raise ValueError("人物一致性不能为空")
    if not plot:
        raise ValueError("故事概括不能为空")
    count = normalize_node_count(node_count)
    tag_system = render_system_prompt(load_text(prompt_path_for(PROMPT_STYLE_NATURAL)), count)
    character_path, summary_path = write_nl_drafts(novel_path, bible, plot)
    nodes, usage_tag = run_chat_step(
        step_name="根据当前人物一致性和故事概括出 TAG（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=tag_system,
        user_prompt=build_nl_tag_user_prompt(bible, plot, novel_path.name, count),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    usages = list(prior_usages or []) + [usage_tag]
    usage = merge_usage(*usages)
    content = f"{bible}\n\n{plot}\n\n{nodes.strip()}"
    report = evaluate_nl_pipeline(bible, plot, nodes, count)
    extra_header = [
        f"- 流水线: `{pipeline_label}`",
        f"- 人物一致性稿: `{character_path}`",
        f"- 故事概括稿: `{summary_path}`",
    ]
    if prior_usages:
        for index, step_usage in enumerate(prior_usages, start=1):
            extra_header.append(
                f"- 第{index}步 prompt/completion: "
                f"{step_usage.get('prompt_tokens', '未知')} / "
                f"{step_usage.get('completion_tokens', '未知')}"
            )
        extra_header.append(
            f"- 第三步 prompt/completion: "
            f"{usage_tag.get('prompt_tokens', '未知')} / "
            f"{usage_tag.get('completion_tokens', '未知')}"
        )
    else:
        extra_header.append(
            f"- 本步 prompt/completion: "
            f"{usage_tag.get('prompt_tokens', '未知')} / "
            f"{usage_tag.get('completion_tokens', '未知')}"
        )
    emit_progress(on_progress, "正在写入结果文件...")
    output_path = write_output(
        content=content,
        novel_path=novel_path,
        model=model,
        usage=usage,
        report=report,
        prompt_style=PROMPT_STYLE_NATURAL,
        node_count=count,
        extra_header=extra_header,
    )
    return {
        "content": content,
        "usage": usage,
        "report": report,
        "output_path": output_path,
        "truncated": truncated,
        "novel_chars": novel_chars,
        "system_chars": extra_system_chars + len(tag_system),
        "prompt_style": PROMPT_STYLE_NATURAL,
        "node_count": count,
        "pipeline": "nl_step3",
        "character_bible": bible,
        "summary": plot,
        "nodes": nodes,
        "character_path": character_path,
        "summary_path": summary_path,
        "step_usages": usages,
    }


def generate_nl_three_step_snapshot(
    *,
    novel_path: Path,
    novel: str,
    truncated: bool,
    api_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    node_count: int,
    on_progress: Callable[[str], None] | None = None,
    on_step_result: Callable[[str, str], None] | None = None,
    api_backend: str | None = None,
) -> dict:
    count = normalize_node_count(node_count)
    character_system = render_system_prompt(load_text(NL_CHARACTER_PROMPT_PATH))
    summary_system = render_system_prompt(load_text(NL_SUMMARY_PROMPT_PATH))

    character_bible, usage_1 = run_chat_step(
        step_name="第 1/3 步：从小说提取人物一致性（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=character_system,
        user_prompt=build_nl_character_user_prompt(novel, novel_path.name),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    write_nl_character_draft(novel_path, character_bible)
    if on_step_result is not None:
        on_step_result("character", character_bible)

    summary, usage_2 = run_chat_step(
        step_name="第 2/3 步：概括小说内容（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=summary_system,
        user_prompt=build_nl_summary_user_prompt(novel, novel_path.name),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    write_nl_summary_draft(novel_path, summary)
    if on_step_result is not None:
        on_step_result("summary", summary)

    result = generate_nl_tag_from_materials(
        novel_path=novel_path,
        character_bible=character_bible,
        summary=summary,
        api_url=api_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        node_count=count,
        on_progress=on_progress,
        pipeline_label="自然语言三步独立对话",
        prior_usages=[usage_1, usage_2],
        truncated=truncated,
        novel_chars=len(novel),
        extra_system_chars=len(character_system) + len(summary_system),
        api_backend=api_backend,
    )
    result["pipeline"] = "nl_three_step"
    return result


def generate_snapshot(
    *,
    novel_path: Path,
    api_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    novel_max_chars: int = DEFAULT_NOVEL_MAX_CHARS,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
    on_progress: Callable[[str], None] | None = None,
    on_step_result: Callable[[str, str], None] | None = None,
    api_backend: str | None = None,
) -> dict:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    novel, truncated = clip_novel(load_text(novel_path), novel_max_chars)
    if not novel:
        raise ValueError(f"小说文件是空的: {novel_path}")
    if style == PROMPT_STYLE_NATURAL:
        return generate_nl_three_step_snapshot(
            novel_path=novel_path,
            novel=novel,
            truncated=truncated,
            api_url=api_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            node_count=count,
            on_progress=on_progress,
            on_step_result=on_step_result,
            api_backend=api_backend,
        )

    system_prompt = render_system_prompt(load_text(prompt_path_for(style)), count)
    user_prompt = build_user_prompt(novel, novel_path.name, style, count)
    emit_progress(on_progress, "正在调用接口生成节点...")
    result = chat_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        api_backend=api_backend,
    )
    content = extract_content(result)
    usage = normalize_usage(result.get("usage") or {})
    report = evaluate(content, style, count)
    output_path = write_output(
        content=content,
        novel_path=novel_path,
        model=model,
        usage=usage,
        report=report,
        prompt_style=style,
        node_count=count,
    )
    return {
        "content": content,
        "usage": usage,
        "report": report,
        "output_path": output_path,
        "truncated": truncated,
        "novel_chars": len(novel),
        "system_chars": len(system_prompt),
        "prompt_style": style,
        "node_count": count,
        "pipeline": "single",
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
