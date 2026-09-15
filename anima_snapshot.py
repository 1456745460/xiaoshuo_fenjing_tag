#!/usr/bin/env python3
"""小说节点快照：读取 txt，调用 OpenAI 兼容接口生成 Anima / 自然语言 / Krea2 提示词。"""

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
DEFAULT_NOVEL_PATH = ROOT / "samples" / "snapshot_test_excerpt.txt"
OUTPUT_PATH = ROOT / "test_output_deepseek.md"
OUTPUT_DIR = ROOT / "outputs"
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


def render_system_prompt(text: str, node_count: int) -> str:
    count = normalize_node_count(node_count)
    if NODE_COUNT_PLACEHOLDER not in text:
        raise ValueError(f"系统提示词缺少 {NODE_COUNT_PLACEHOLDER} 占位符")
    return text.replace(NODE_COUNT_PLACEHOLDER, str(count))


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
            "每个节点的中文提示词和英文提示词必须是完整自然语言画面描述，"
            "可直接用于自然语言文生图。禁止输出 Danbooru tag、snake_case、逗号堆砌标签。"
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
    found_count = max(len(node_hits), len(cn_hits), len(en_hits))
    lines = [
        f"提示词类型: {PROMPT_STYLE_LABELS[style]}",
        f"要求分镜数: {expected}",
        f"角色档案: {'有' if has_bible else '缺失'}",
        f"节点标题数: {len(node_hits)}",
        f"中文提示词块: {len(cn_hits)}",
        f"英文提示词块: {len(en_hits)}",
        f"禁用质量套话: {quality_hits or '无'}",
    ]
    structure_ok = has_bible and found_count == expected and not quality_hits
    if style in (PROMPT_STYLE_NATURAL, PROMPT_STYLE_KREA2):
        lines.append(f"Danbooru tag 残留: {len(tag_hits)}")
        ok = structure_ok
        if len(tag_hits) >= 8:
            target = "Krea2 导演简报" if style == PROMPT_STYLE_KREA2 else "自然语言"
            lines.append(f"警告: 正文里仍出现较多 Danbooru tag，请核对是否按{target}输出")
            ok = False
        if style == PROMPT_STYLE_KREA2 and "全书视觉锁定" not in content:
            lines.append("警告: 未找到全书视觉锁定，Krea2 容易漂风格")
            ok = False
    else:
        lines.append(f"关键 Anima tag 命中: {len(tag_hits)}")
        ok = structure_ok and len(tag_hits) >= 8
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
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
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
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    node_count: int = DEFAULT_NODE_COUNT,
) -> dict:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    system_prompt = render_system_prompt(load_text(prompt_path_for(style)), count)
    novel, truncated = clip_novel(load_text(novel_path), novel_max_chars)
    if not novel:
        raise ValueError(f"小说文件是空的: {novel_path}")
    user_prompt = build_user_prompt(novel, novel_path.name, style, count)
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
