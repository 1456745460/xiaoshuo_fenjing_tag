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
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")
STAGE_CHARACTER_NAME = "1_character.md"
STAGE_SUMMARY_NAME = "2_summary.md"
STAGE_TAGS_NAME = "3_tags.md"
STAGE_SNAPSHOT_NAME = "snapshot.md"
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
CLOTHING_MODE_PLACEHOLDER = "{{CLOTHING_MODE}}"
DEFAULT_SEXY_CLOTHING = False
SEXY_CLOTHING_RULE = """【本局已开启「性感服装」】此条覆盖后文所有「服装必须跟正文 / 档案 / 概括」的条款，包括「正文写到的服装禁止改写」「档案默认服装必须继承」。
- 发色、瞳色、体型、年龄、姿势、场景、场次仍按正文/档案/概括；只有服装整条链路改写。
- 禁止沿用故事里的常服、职业装、校服、家居服、睡衣原样。正文或档案写衬衫/裙子/制服/睡衣，也必须丢掉原装，另做更性感的一套。
- 必须大胆猜想：为每个可入画角色重新设计更暴露、更能拉开视觉差的服装。禁止因为「正文写了白衬衫」就继续白衬衫。
- 每套服装必须写全：品类、颜色、暴露状态、花纹、样式、点缀、面料、剪裁。禁止只写「性感」或只写品类。服装全部记入推断项，不要假装来自原文。
- 优先叠加短、紧、薄、透、低胸、露腰、露背、高开衩、吊带、超短裙、贴身皮/漆皮、蕾丝、渔网、吊带袜、细高跟、内衣外穿、湿身贴体、半脱露肤。不同角色必须错开颜色、品类、花纹、点缀，禁止全员同一套黑丝御姐。
- 颜色大胆：酒红、黑、深紫、珊瑚粉、香槟金、纯白蕾丝、暗红碎花、墨绿、肤色薄纱。禁止全员白衬衫黑裤。
- 花纹/样式/点缀必须写：碎花、条纹、波点、蕾丝花边、蝴蝶结、金属扣、交叉绑带、颈环、腰链、吊坠、开窗、荷叶边、镂空、侧开衩。
- 暴露状态必须写清：哪些部位被布料覆盖、哪些被剪裁露出（锁骨、乳沟、腰腹、大腿根、臀线、腋下、后背、侧乳）。情爱节点仍按脱衣顺序更新「脱掉哪件、落到哪里、还剩什么」，但脱的是这套性感服装，不是故事原装。
- 可保留场景时代的影子（现代都市不要盔甲，古风不要西装），服装本身必须重新大胆设计。
- 同一角色全节点沿用这套性感默认装，只随脱衣/凌乱更新状态。
- 男性也要更性感：更贴身、更低腰、敞开领口、更少布料或更暴露的穿着状态；不要给男性套女装，除非正文就是女装。
- 禁止用 beautiful / sexy 这种空词代替造型。品类仍从服装语法卡选，可叠加露、透、短、紧、蕾丝、渔网、吊带。
- 性感向品类优先：crop top / camisole / bustier / lace bralette / off-shoulder blouse；micro miniskirt / high-slit dress / micro shorts；lingerie / babydoll / sheer nightgown / garter belt；fishnets / lace thighhighs / stay-ups / stiletto heels；点缀 choker / waist chain / ribbon / lace trim / cross straps。
"""
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


def normalize_sexy_clothing(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return DEFAULT_SEXY_CLOTHING
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on", "开", "是"}:
        return True
    if text in {"", "0", "false", "no", "off", "关", "否"}:
        return False
    raise ValueError("性感服装开关必须是开或关")


def clothing_mode_rule(sexy_clothing: bool) -> str:
    return SEXY_CLOTHING_RULE.strip() if sexy_clothing else ""


def clothing_user_instruction(sexy_clothing: bool) -> str:
    if sexy_clothing:
        return (
            "本局已开启「性感服装」：禁止沿用故事/档案/概括里的原装。"
            "必须大胆猜想更性感的服装，写全品类、颜色、暴露状态、花纹、样式、点缀、面料、剪裁。"
            "脱衣只更新这套性感装的状态。发色、瞳色、体型、姿势、场次仍按原文/档案/概括。"
        )
    return (
        "服装必须写颜色、花纹、面料纹理、剪裁；品类从服装语法卡选，禁止只写衬衫/裙子/white_shirt。"
    )


def apply_clothing_mode(text: str, sexy_clothing: bool = False) -> str:
    enabled = normalize_sexy_clothing(sexy_clothing)
    block = clothing_mode_rule(enabled)
    if CLOTHING_MODE_PLACEHOLDER in text:
        if block:
            return text.replace(CLOTHING_MODE_PLACEHOLDER, block + "\n")
        return text.replace(CLOTHING_MODE_PLACEHOLDER + "\n", "").replace(
            CLOTHING_MODE_PLACEHOLDER, ""
        )
    if enabled and block:
        return block + "\n\n" + text
    return text


def render_system_prompt(
    text: str,
    node_count: int | None = None,
    sexy_clothing: bool = False,
) -> str:
    rendered = text
    if NODE_COUNT_PLACEHOLDER in rendered:
        if node_count is None:
            raise ValueError(f"系统提示词含 {NODE_COUNT_PLACEHOLDER}，但未提供分镜数")
        rendered = rendered.replace(
            NODE_COUNT_PLACEHOLDER, str(normalize_node_count(node_count))
        )
    return apply_clothing_mode(rendered, sexy_clothing)


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
    sexy_clothing: bool = False,
) -> str:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    if style == PROMPT_STYLE_KREA2:
        prompt_rule = (
            "每个节点的英文提示词必须是可直接贴进 Krea 2 的导演简报，80 到 160 词。"
            "必须先写全书风格锁定句（2D anime still frame, modern TV anime cel, visible linework, cel-shaded 等），再写机位/人数，再按人写锁定外貌+此刻服装动作，再写场景、主光、赛璐璐阴影。"
            "禁止 photoreal / live-action / cinematic still / 35mm / film grain。禁止 Danbooru tag、snake_case、BREAK、1girl、solo。中文提示词信息对齐，供阅读；出图只贴英文。"
            "人数用肯定句：frame holds only N people（N≤3）。不要堆 no crowd / no bystanders 否定清单。"
            "单人、双人禁止 On the left / On the right / Close on the right，左右站位只给三人。"
            "每个分镜最多 3 个可识别人物；正文超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
            "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把人数加一。"
            "多人必须叙事构图：过肩/侧面/面对面，写清谁看谁，禁止并排看镜头，禁止 looking at the camera。"
            "表情写五官动作；女性爱按强度用愉悦/高潮；男色欲用邪笑/得意/坏笑，不要乱加。"
        )
    elif style == PROMPT_STYLE_NATURAL:
        prompt_rule = (
            "中文提示词必须是逗号分隔的画面要素，只写看得见的外形、服装、姿势、道具，禁止小说修辞。"
            "英文必须是 Anima 混合写法："
            "标签块（masterpiece, best quality, score_7 + 1girl/1boy + 视线互动 + 按人外观服装）空一行后再写短句。"
            "禁止英文散文、snake_case、BREAK、(word:1.2)、looking_at_viewer。"
            "禁止憨痴惹人爱怜、雪白细嫩、斯文端正、眼含雾、misty、sweet naive、delicate、rim light。"
            "多人同框必须一人一句，主语带外貌锚点，独有特征禁止写成全画面清单。"
            "段首先锁人数：画面里只有 N 个人（N≤3）。每个分镜最多 3 个可识别人物；正文超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
            "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把两人写成三人、三人写成四人。"
            "多人必须叙事构图：面对面或侧面相对。人数 tag 后立刻写 looking at another / looking at each other / eye contact，"
            "双人短句写谁看谁（the woman is looking at the boy）；三人才写 the girl on the left is looking at the boy。"
            "视线禁止留空，禁止并排看镜头合影，禁止 looking at the viewer。"
            "左右身份句只给三人：On the left, boy / On the right, lean young man，站位后逗号，身份前禁止 a / an / the。"
            "单人、双人直接写身份，禁止 on the left / on the right / 画面左侧 / 画面右侧，Anima 会把人数算错。"
            "表情写五官动作，禁止只写微笑/脸红；女性爱按强度用愉悦表情/高潮脸/阿嘿颜等，男色欲用邪笑/得意/坏笑，不要乱加。"
        )
    else:
        prompt_rule = (
            "每个节点的中文提示词和英文提示词必须完整、可直接用于 Anima 文生图。"
            "英文必须是 Danbooru / snake_case tag，逗号分隔，禁止写成自然语言句子。"
            "多人同框必须用 BREAK 分角色块，独有特征禁止写进全局段，并写清互斥项。"
            "单人、双人禁止 on_the_left / on_the_right / in_the_center；左右站位只给三人。"
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
        "单人、双人禁止左右站位：不要写 on the left / on the right / on_the_left / on_the_right / 画面左/右侧。左右身份句只给三人。\n"
        "防人数膨胀：每个分镜最多 3 个可识别人物，禁止 4 人同框；在场几人就只写几人；玻璃/镜子只写光斑不写人物倒影；锁定外貌禁止 married/妻子/丈夫；走廊办公室默认空场。\n"
        "防合影：多人禁止看镜头，必须对视或看对方身体，机位用侧面/过肩/面对面，不要正面并排。\n"
        "表情：写五官动作；女性爱按强度选愉悦/高潮/阿嘿颜；男色欲用邪笑/得意/坏笑，禁止一律微笑，禁止乱加。\n"
        "角色必须一眼能分清：正文没写死时，不同人要错开发色、长短发、身材、胡子/眼镜/痣；禁止全员同一张脸。\n"
        f"{clothing_user_instruction(sexy_clothing)}\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def build_nl_character_user_prompt(
    novel: str,
    novel_name: str,
    sexy_clothing: bool = False,
) -> str:
    return (
        "请根据下面这篇 txt 小说正文，只产出「角色一致性档案」。\n"
        "锁定中英文外貌必须是逗号分隔的外形词，只写看得见的发、瞳、脸型、身材、皮肤、眼镜。\n"
        "禁止憨痴惹人爱怜、甜相、雪白细嫩、斯文端正、眼含雾、misty、sweet naive、delicate。\n"
        "不要写节点，不要写中英文提示词，不要写剧情摘要。不要寒暄。\n"
        f"{clothing_user_instruction(sexy_clothing)}\n\n"
        f"小说文件名：{novel_name}\n\n"
        "<novel>\n"
        f"{novel}\n"
        "</novel>"
    )


def build_nl_summary_user_prompt(
    novel: str,
    novel_name: str,
    sexy_clothing: bool = False,
) -> str:
    extra = ""
    if sexy_clothing:
        extra = (
            "本局已开启「性感服装」：姿势、场次、脱衣顺序仍按正文；"
            "服装本身不要跟故事原装，改写成更性感的一套，并按这套性感装写脱衣/还剩什么。\n"
        )
    return (
        "请根据下面这篇 txt 小说正文，写出保留完整故事结构的详细概括。\n"
        "不要过于精简。严格保留整体故事结构、不同爱情姿势、不同爱情经过。\n"
        "每一场情爱单独写，每换一种姿势就分条写，禁止合并成「两人做爱」。\n"
        "不要写角色外貌档案，不要写分镜提示词。不要寒暄。\n"
        f"{extra}\n"
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
    sexy_clothing: bool = False,
) -> str:
    count = normalize_node_count(node_count)
    prompt_rule = (
        "中文提示词必须是逗号分隔的画面要素，一人一截，主语带外形词。"
        "只写看得见的外形、服装、姿势、五官动作、道具。禁止小说修辞。"
        "禁止憨痴惹人爱怜、甜相、极为饱满、圆润柔和、雪白细嫩、斯文端正、眼含雾、蓝光侧照。"
        "英文提示词必须是 Anima 混合写法：先写小写空格分词的标签块"
        "（masterpiece, best quality, score_7，NSFW 加 explicit，再写 1girl/1boy 和按人分行的外观服装），"
        "空一行后写短英文：单人 3 到 4 句（构图、看哪里、此人、场景道具），"
        "双人 4 到 6 句（构图、谁看谁、此人、彼人、场景道具、归属），"
        "三人 4 到 6 句（构图、谁看谁、左、右、场景道具、归属）。"
        "标签用 round face / large breasts / pale skin，禁止 misty / sweet naive / delicate / rim light。"
        "禁止把英文写成一篇嵌套从句散文。禁止 snake_case、BREAK、(word:1.2)、looking_at_viewer。"
        "锁定外形词写进英文标签段和中文提示词；短句只回锚 2～4 个辨识点，不要把文艺锁定句整段嵌进从句。"
        "段首先锁人数：画面里只有 N 个人（N≤3）。每个分镜最多 3 个可识别人物；"
        "概括超过 3 人必须拆镜或只留核心 1～3 人，禁止 4 人同框。"
        "禁止用玻璃倒影、married/丈夫/妻子、走廊路人把两人写成三人、三人写成四人。"
        "单人、双人禁止防串站位：禁止 on the left / on the right / 画面左侧 / 画面右侧，Anima 会把人数算错。"
        "多人必须叙事构图：面对面或侧面相对。人数 tag 后立刻写 looking at another / looking at each other / eye contact，"
        "双人短句写谁看谁（the woman is looking at the boy）；三人才写 the girl on the left is looking at the boy。"
        "视线禁止留空，禁止并排看镜头合影，禁止 looking at the viewer。"
        "左右身份句只给三人，写成 On the left, boy / On the right, lean young man："
        "站位后立刻逗号，身份用无冠词标签，禁止 On the left a young woman / On the right a lean young man。"
        "单人、双人直接写 Young woman, [action] / Lean young man, [action]。"
        "表情写五官动作，禁止只写微笑/脸红；女性爱按强度用愉悦表情/高潮脸/阿嘿颜等，"
        "男色欲用邪笑/得意/坏笑，不要乱加。"
    )
    return (
        "请根据下面已经完成的「角色一致性档案」和「小说内容概括」，严格执行系统提示词。\n"
        f"本次必须输出正好 {count} 个关键节点，不多不少。状态表写要点即可，"
        f"但{prompt_rule}\n"
        "不要再读原文，不要重写发色瞳色体型脸型。锁定句若是文艺描写，先收成短外形词再写入，禁止原样粘贴。\n"
        "不要再输出角色一致性档案，不要再输出小说概括。直接从节点目录写起。\n"
        "概括里的不同爱情姿势、不同爱情经过必须尽量用不同节点覆盖，禁止合并省略。\n"
        "防串台：禁止把发色、眼镜、服装混成一袋；一人戴眼镜则另一人必须明确不戴。\n"
        "单人、双人禁止左右站位：不要写 on the left / on the right / 画面左/右侧。\n"
        "防人数膨胀：每个分镜最多 3 个可识别人物，禁止 4 人同框；在场几人就只写几人；"
        "玻璃/镜子只写光斑不写人物倒影；锁定外貌禁止 married/妻子/丈夫；走廊办公室默认空场。\n"
        "防合影：多人禁止看镜头。人数 tag 后立刻写 looking at another / looking at each other，"
        "短句写谁看谁，机位用侧面/过肩/面对面，不要正面并排。\n"
        "三人左右身份句：On the left, boy / On the right, lean young man。站位后逗号，禁止 a / an。单人、双人直接写身份。\n"
        "表情：写五官动作；女性爱按强度选愉悦/高潮/阿嘿颜；男色欲用邪笑/得意/坏笑，禁止一律微笑，禁止乱加。\n"
        f"{clothing_user_instruction(sexy_clothing)}\n"
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
        prompt_blob = "\n".join(
            match.group(1)
            for match in re.finditer(
                r"(?:中文提示词|英文提示词)[：:](.*?)(?=\n- |\n#{1,3}\s|\Z)",
                content,
                flags=re.S,
            )
        )
        literary_hits = _LITERARY_WARN_RE.findall(prompt_blob)
        if literary_hits:
            lines.append(
                f"警告: {len(literary_hits)} 处文艺修辞残留（憨痴惹人爱怜/雪白细嫩/misty/sweet naive 等），应改成圆脸/大胸/白皮肤等外形词"
            )
            ok = False
        article_hits = re.findall(
            r"(?im)(?:^|[.\n]\s*)on the (?:left|right),?\s+an?\s+",
            content,
        )
        if article_hits:
            lines.append(
                f"警告: {len(article_hits)} 处三人左右身份句仍带 a/an，应为 On the left, boy / On the right, lean young man"
            )
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
    under_three_pos = sum(
        1
        for block in _NODE_BLOCK_RE.split(content)[1:]
        if _node_person_count(block) in {1, 2} and _node_has_positioning(block)
    )
    if under_three_pos:
        lines.append(
            f"警告: {under_three_pos} 个单人/双人节点含 on the left / on the right，左右站位只给三人"
        )
        ok = False
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


_ERECT_PENIS_RE = re.compile(r"erect penis", re.IGNORECASE)
_NODE_BLOCK_RE = re.compile(r"(?=^(?:#{1,3}\s*)?节点\s+\d+)", re.M)
_CAST_LINE_RE = re.compile(r"^-\s*在场人物[：:].+$", re.M)
_CAST_ID_RE = re.compile(r"C\d+", re.I)
_COUNT_TAG_RE = re.compile(r"\b(1girl|1boy|2girls|3girls|2boys|3boys)\b", re.I)
_LENS_SOLO_RE = re.compile(r"^-\s*镜头[：:].*单人", re.M)
_FANGCHUAN_LINE_RE = re.compile(r"^-\s*防串站位[：:].+$", re.M)
_SOLO_POS_EN_RE = re.compile(
    r"(?:,\s*)?(?:"
    r"\bon[ _-]the[ _-](?:left|right)(?: of the frame)?\b"
    r"|\bin the cent(?:er|re)[ -](?:left|right)\b"
    r"|\bcent(?:er|re)[ -](?:left|right)\b"
    r"|\bclose on the (?:left|right)\b"
    r")",
    re.I,
)
_SOLO_POS_CN_RE = re.compile(
    r"(?:位于)?画面(?:的)?(?:左侧|右侧|中央偏左|中央偏右)"
    r"|(?:位于)?中央偏左"
    r"|中央偏右"
)
_SOLO_FANGCHUAN_REPL = "- 防串站位：单人无需防串站位"
_DUO_FANGCHUAN_REPL = "- 防串站位：双人无需左右站位"
_LENS_DUO_RE = re.compile(r"^-\s*镜头[：:].*双人", re.M)
_LENS_TRIO_RE = re.compile(r"^-\s*镜头[：:].*三人", re.M)
_ON_SIDE_IDENTITY_RE = re.compile(
    r"(?im)(^|[.\n][ \t]*)On the (?P<side>left|right)"
    r"(?:[ \t]+of the frame)?"
    r"(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?:(?P<article>an?|the)[ \t]+)?"
)


def _node_person_count(block: str) -> int | None:
    cast_match = _CAST_LINE_RE.search(block)
    if cast_match:
        ids = {item.lower() for item in _CAST_ID_RE.findall(cast_match.group(0))}
        if ids:
            return min(len(ids), 4)
    if _LENS_SOLO_RE.search(block):
        return 1
    if _LENS_DUO_RE.search(block):
        return 2
    if _LENS_TRIO_RE.search(block):
        return 3
    tags = {item.lower() for item in _COUNT_TAG_RE.findall(block)}
    if not tags:
        return None
    girl_n = 3 if "3girls" in tags else 2 if "2girls" in tags else 1 if "1girl" in tags else 0
    boy_n = 3 if "3boys" in tags else 2 if "2boys" in tags else 1 if "1boy" in tags else 0
    total = girl_n + boy_n
    if total:
        return min(total, 4)
    return None


def _node_has_positioning(block: str) -> bool:
    return bool(_SOLO_POS_EN_RE.search(block) or _SOLO_POS_CN_RE.search(block))


def _cleanup_stripped_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" ?, ?,+", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"(?m)^,\s*", "", text)
    return text


def _strip_under_three_positioning(block: str, person_count: int) -> str:
    fangchuan = _SOLO_FANGCHUAN_REPL if person_count == 1 else _DUO_FANGCHUAN_REPL
    block = _FANGCHUAN_LINE_RE.sub(fangchuan, block, count=1)
    block = _SOLO_POS_EN_RE.sub("", block)
    block = _SOLO_POS_CN_RE.sub("", block)
    return _cleanup_stripped_text(block)


def _normalize_on_side_identity(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}On the {match.group('side').lower()}, "

    return _ON_SIDE_IDENTITY_RE.sub(repl, text)


_CN_LITERARY_REPLACEMENTS = (
    ("憨痴惹人爱怜的甜相", "可爱脸"),
    ("憨痴惹人爱怜的少女感", "年轻"),
    ("憨痴惹人爱怜", "可爱"),
    ("憨痴甜相", "可爱脸"),
    ("惹人爱怜", ""),
    ("极为饱满挺翘的大胸", "大胸"),
    ("饱满挺翘的大胸", "大胸"),
    ("极为饱满挺翘", "大胸"),
    ("圆润柔和的脸型", "圆脸"),
    ("雪白细嫩的皮肤", "白皮肤"),
    ("雪白细嫩肤质", "白皮肤"),
    ("皮肤雪白细腻", "白皮肤"),
    ("雪白细嫩", "白皮肤"),
    ("雪白细腻", "白皮肤"),
    ("斯文端正刮净的脸", "刮净脸"),
    ("轮廓分明刮净的脸", "刮净脸"),
    ("清秀端正的脸", "刮净脸"),
    ("斯文端正", ""),
    ("清秀端正", ""),
    ("眼含雾", "半睁眼"),
    ("眼含春", "半睁眼"),
    ("脸颊强烈潮红", "脸红"),
    ("脸颊飞红", "脸红"),
    ("脸颊潮红", "脸红"),
    ("脸颊泛红", "脸红"),
    ("脸颊羞红", "脸红"),
    ("脸颊通红", "脸红"),
    ("粉臀轻摆", "臀摆动"),
    ("小嘴喘气", "张嘴喘气"),
    ("唇大张浪叫", "张嘴"),
    ("浅浅淡淡的阴毛", "稀疏阴毛"),
    ("浅浅淡淡阴毛", "稀疏阴毛"),
    ("似笑非笑", "嘴角微扬"),
    ("薄怒", "微皱眉"),
    ("乳房丰满坚挺", "大胸"),
    ("屁股圆润上翘", "圆臀"),
    ("高翘圆臀", "圆臀"),
    ("双腿修长笔直", "长腿"),
    ("体态丰满圆润", "丰满身材"),
    ("嘴角上挑露出淫笑", "嘴角上扬"),
    ("唇瓣微张", "嘴唇微张"),
    ("眉头轻蹙", "微皱眉"),
    ("眉头紧蹙", "皱眉"),
    ("心不在焉", ""),
    ("电视蓝光侧照", "电视"),
    ("橙红夕照侧逆光", "傍晚"),
    ("暮色蓝调侧光", "傍晚"),
    ("夜色冷光侧照", "夜"),
    ("床头灯暖黄侧光", "台灯"),
    ("路灯昏黄侧光", "路灯"),
    ("路灯余光侧照", "路灯"),
    ("前景虚化", ""),
    ("成熟妩媚", "成熟脸"),
)
_EN_LITERARY_REPLACEMENTS = (
    ("a sweet naive look", "innocent face"),
    ("sweet naive look", "innocent face"),
    ("misty black eyes", "black eyes"),
    ("very full and perky large breasts", "large breasts"),
    ("very full perky large breasts", "large breasts"),
    ("round soft facial features", "round face"),
    ("round soft face", "round face"),
    ("round high buttocks", "round ass"),
    ("fair delicate skin", "pale skin"),
    ("a mature alluring look", "mature face"),
    ("mature alluring look", "mature face"),
    ("eyes misty", "eyes half closed"),
    ("rim light from the side", ""),
    ("warm yellow bedside lamp light from the side", "lamp"),
    ("warm yellow lamp light from the side", "lamp"),
    ("orange sunset rim light from the side", "dusk"),
    ("cool blue dusk side light", "dusk"),
    ("unaware of the camera, candid", "not looking at camera"),
    ("unaware of the camera", "not looking at camera"),
)
_LITERARY_WARN_RE = re.compile(
    r"憨痴惹人爱怜|雪白细嫩|斯文端正|圆润柔和|眼含雾|sweet naive|misty black|fair delicate|round soft face",
    re.I,
)


def _flatten_literary_phrasing(text: str) -> str:
    for src, dst in _CN_LITERARY_REPLACEMENTS:
        text = text.replace(src, dst)
    for src, dst in _EN_LITERARY_REPLACEMENTS:
        text = re.sub(re.escape(src), dst, text, flags=re.I)
    return _cleanup_stripped_text(text)


def postprocess_step3_tags(text: str, prompt_style: str | None = None) -> str:
    text = _ERECT_PENIS_RE.sub("huge penis", text)
    parts = _NODE_BLOCK_RE.split(text)
    if len(parts) > 1:
        rewritten = [parts[0]]
        for block in parts[1:]:
            person_count = _node_person_count(block)
            if person_count in {1, 2}:
                rewritten.append(_strip_under_three_positioning(block, person_count))
            else:
                rewritten.append(block)
        text = "".join(rewritten)
    style = (
        normalize_prompt_style(prompt_style)
        if prompt_style
        else PROMPT_STYLE_NATURAL
    )
    if style == PROMPT_STYLE_NATURAL:
        text = _normalize_on_side_identity(text)
        text = _flatten_literary_phrasing(text)
    return text


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


def make_run_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = OUTPUT_DIR / stamp
    extra = 1
    while target.exists():
        extra += 1
        target = OUTPUT_DIR / f"{stamp}_{extra}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def stage_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / STAGE_CHARACTER_NAME,
        run_dir / STAGE_SUMMARY_NAME,
        run_dir / STAGE_TAGS_NAME,
    )


def list_run_dirs() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    dirs = [
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and RUN_DIR_RE.match(path.name)
    ]
    return sorted(dirs, key=lambda path: path.name, reverse=True)


def _run_dir_matches_novel(run_dir: Path, novel_path: Path) -> bool:
    _, _, tags_path = stage_paths(run_dir)
    if not tags_path.exists():
        return False
    try:
        head = tags_path.read_text(encoding="utf-8")[:4000]
    except OSError:
        return False
    stem = novel_path.stem
    return stem in head or str(novel_path) in head


def find_latest_run_dir(novel_path: Path | None = None) -> Path | None:
    unmatched_drafts: Path | None = None
    for run_dir in list_run_dirs():
        character_path, summary_path, tags_path = stage_paths(run_dir)
        has_drafts = character_path.exists() and summary_path.exists()
        if novel_path is not None and tags_path.exists():
            if _run_dir_matches_novel(run_dir, novel_path):
                return run_dir
            continue
        if has_drafts and unmatched_drafts is None:
            unmatched_drafts = run_dir
    return unmatched_drafts


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
    run_dir: Path | None = None,
    sexy_clothing: bool = False,
) -> Path:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    sexy = normalize_sexy_clothing(sexy_clothing)
    if output_path is None:
        dest = run_dir or make_run_dir()
        dest.mkdir(parents=True, exist_ok=True)
        filename = STAGE_TAGS_NAME if style == PROMPT_STYLE_NATURAL else STAGE_SNAPSHOT_NAME
        target = dest / filename
    else:
        target = output_path
        target.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# 小说节点快照",
        "",
        f"- 模型: `{model}`",
        f"- 小说: `{novel_path}`",
        f"- 提示词类型: `{PROMPT_STYLE_LABELS[style]}` ({style})",
        f"- 分镜数: `{count}`",
        f"- 性感服装: {'开' if sexy else '关'}",
        f"- 输出目录: `{target.parent}`",
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


def nl_draft_paths(
    novel_path: Path,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    dest = run_dir or find_latest_run_dir(novel_path)
    if dest is not None:
        character_path, summary_path, _ = stage_paths(dest)
        return character_path, summary_path
    stem = novel_path.stem
    return (
        INTERMEDIATE_DIR / f"{stem}_character.md",
        INTERMEDIATE_DIR / f"{stem}_summary.md",
    )


def write_nl_character_draft(
    novel_path: Path,
    character_bible: str,
    run_dir: Path | None = None,
) -> Path:
    dest = run_dir or make_run_dir()
    character_path, _, _ = stage_paths(dest)
    dest.mkdir(parents=True, exist_ok=True)
    character_path.write_text(character_bible.strip() + "\n", encoding="utf-8")
    return character_path


def write_nl_summary_draft(
    novel_path: Path,
    summary: str,
    run_dir: Path | None = None,
) -> Path:
    dest = run_dir or make_run_dir()
    _, summary_path, _ = stage_paths(dest)
    dest.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.strip() + "\n", encoding="utf-8")
    return summary_path


def write_nl_drafts(
    novel_path: Path,
    character_bible: str,
    summary: str,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    dest = run_dir or make_run_dir()
    return (
        write_nl_character_draft(novel_path, character_bible, run_dir=dest),
        write_nl_summary_draft(novel_path, summary, run_dir=dest),
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
    run_dir: Path | None = None,
    sexy_clothing: bool = False,
) -> dict:
    bible = character_bible.strip()
    plot = summary.strip()
    if not bible:
        raise ValueError("人物一致性不能为空")
    if not plot:
        raise ValueError("故事概括不能为空")
    count = normalize_node_count(node_count)
    sexy = normalize_sexy_clothing(sexy_clothing)
    tag_system = render_system_prompt(
        load_text(prompt_path_for(PROMPT_STYLE_NATURAL)),
        count,
        sexy_clothing=sexy,
    )
    dest = run_dir or make_run_dir()
    character_path, summary_path = write_nl_drafts(novel_path, bible, plot, run_dir=dest)
    nodes, usage_tag = run_chat_step(
        step_name="根据当前人物一致性和故事概括出 TAG（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=tag_system,
        user_prompt=build_nl_tag_user_prompt(
            _flatten_literary_phrasing(bible),
            plot,
            novel_path.name,
            count,
            sexy_clothing=sexy,
        ),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    nodes = postprocess_step3_tags(nodes, PROMPT_STYLE_NATURAL)
    usages = list(prior_usages or []) + [usage_tag]
    usage = merge_usage(*usages)
    content = f"{bible}\n\n{plot}\n\n{nodes.strip()}"
    report = evaluate_nl_pipeline(bible, plot, nodes, count)
    extra_header = [
        f"- 流水线: `{pipeline_label}`",
        f"- 人物一致性稿: `{character_path}`",
        f"- 故事概括稿: `{summary_path}`",
        f"- 第三步 TAG: `{dest / STAGE_TAGS_NAME}`",
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
        content=nodes.strip(),
        novel_path=novel_path,
        model=model,
        usage=usage,
        report=report,
        prompt_style=PROMPT_STYLE_NATURAL,
        node_count=count,
        extra_header=extra_header,
        run_dir=dest,
        sexy_clothing=sexy,
    )
    return {
        "content": content,
        "usage": usage,
        "report": report,
        "output_path": output_path,
        "run_dir": dest,
        "truncated": truncated,
        "novel_chars": novel_chars,
        "system_chars": extra_system_chars + len(tag_system),
        "prompt_style": PROMPT_STYLE_NATURAL,
        "node_count": count,
        "sexy_clothing": sexy,
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
    sexy_clothing: bool = False,
) -> dict:
    count = normalize_node_count(node_count)
    sexy = normalize_sexy_clothing(sexy_clothing)
    character_system = render_system_prompt(
        load_text(NL_CHARACTER_PROMPT_PATH), sexy_clothing=sexy
    )
    summary_system = render_system_prompt(
        load_text(NL_SUMMARY_PROMPT_PATH), sexy_clothing=sexy
    )
    run_dir = make_run_dir()

    character_bible, usage_1 = run_chat_step(
        step_name="第 1/3 步：从小说提取人物一致性（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=character_system,
        user_prompt=build_nl_character_user_prompt(
            novel, novel_path.name, sexy_clothing=sexy
        ),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    write_nl_character_draft(novel_path, character_bible, run_dir=run_dir)
    if on_step_result is not None:
        on_step_result("character", character_bible)

    summary, usage_2 = run_chat_step(
        step_name="第 2/3 步：概括小说内容（新对话）",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=summary_system,
        user_prompt=build_nl_summary_user_prompt(
            novel, novel_path.name, sexy_clothing=sexy
        ),
        max_tokens=max_tokens,
        on_progress=on_progress,
        api_backend=api_backend,
    )
    write_nl_summary_draft(novel_path, summary, run_dir=run_dir)
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
        run_dir=run_dir,
        sexy_clothing=sexy,
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
    sexy_clothing: bool = False,
) -> dict:
    style = normalize_prompt_style(prompt_style)
    count = normalize_node_count(node_count)
    sexy = normalize_sexy_clothing(sexy_clothing)
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
            sexy_clothing=sexy,
        )

    system_prompt = render_system_prompt(
        load_text(prompt_path_for(style)), count, sexy_clothing=sexy
    )
    user_prompt = build_user_prompt(
        novel, novel_path.name, style, count, sexy_clothing=sexy
    )
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
    content = postprocess_step3_tags(extract_content(result), style)
    usage = normalize_usage(result.get("usage") or {})
    report = evaluate(content, style, count)
    run_dir = make_run_dir()
    output_path = write_output(
        content=content,
        novel_path=novel_path,
        model=model,
        usage=usage,
        report=report,
        prompt_style=style,
        node_count=count,
        run_dir=run_dir,
        sexy_clothing=sexy,
    )
    return {
        "content": content,
        "usage": usage,
        "report": report,
        "output_path": output_path,
        "run_dir": run_dir,
        "truncated": truncated,
        "novel_chars": len(novel),
        "system_chars": len(system_prompt),
        "prompt_style": style,
        "node_count": count,
        "pipeline": "single",
        "sexy_clothing": sexy,
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
