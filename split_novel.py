#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把小说 TXT 智能拆成「一书一文件夹、一章一文件」。

用法:
    python split_novel.py 小说.txt
    python split_novel.py a.txt b.txt --dry-run
    python split_novel.py 小说.txt -o /path/to/output
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "utf-16")

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "○": 0,
    "Ｏ": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# 编号类标题：第1章 / 第一章 / 第001回 标题
NUMBERED_HEADING = re.compile(
    r"^第\s*([0-9０-９零〇○Ｏ一二三四五六七八九十百千万两]+)\s*"
    r"([章节回卷部集篇])"
    r"(?:\s*[、.．:：\-—–~～]?\s*(.*))?$"
)
# Chapter 1 / CHAPTER 12 Title
LATIN_CHAPTER = re.compile(
    r"^(?:Chapter|CHAPTER|chapter)\s*([0-9０-９]+)(?:\s*[、.．:：\-—]?\s*(.*))?$"
)
# 楔子 / 序章 / 后记 / 番外 等无编号结构
SPECIAL_HEADING = re.compile(
    r"^(楔子|序章|序言|前言|引子|引言|开场|尾声|后记|终章|终章感言|"
    r"完本感言|作者的话|番外(?:\s*[0-9０-９零〇一二三四五六七八九十]+)?"
    r"(?:[、.．:：\s　].*)?)$"
)

DIALOGUE_PREFIX = tuple('“"「『【（《(')
INVALID_FILENAME = re.compile(r'[\\/:*?"<>|\r\n\t]+')
TITLE_BREAKERS = ("，", ",", "：", ":")


@dataclass(frozen=True)
class Heading:
    line_index: int
    raw: str
    title: str
    kind: str
    unit: str
    number: Optional[int]


@dataclass
class SplitPlan:
    source: Path
    encoding: str
    lines: List[str]
    headings: List[Heading]
    specials: List[Heading]


def read_text(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    if not data:
        return "", "utf-8"
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def chinese_to_int(token: str) -> Optional[int]:
    token = token.strip().translate(FULLWIDTH_DIGITS)
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if any(ch not in CN_DIGITS and ch not in CN_UNITS for ch in token):
        return None
    total = 0
    current = 0
    for ch in token:
        if ch in CN_DIGITS:
            current = CN_DIGITS[ch]
            continue
        unit = CN_UNITS[ch]
        if current == 0:
            current = 1
        if unit >= 10000:
            total = (total + current) * unit
            current = 0
        else:
            total += current * unit
            current = 0
    return total + current


def normalize_line(line: str) -> str:
    text = line.strip().strip("\ufeff")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip("　 \t")


def is_sentence_title(extra: str) -> bool:
    """正文里常把『第N章，……』写成一句话，标题本身很少这样。"""
    if not extra:
        return False
    if extra.startswith(TITLE_BREAKERS):
        return True
    if extra.count("，") + extra.count(",") >= 1 and len(extra) >= 8:
        return True
    if "说：" in extra or "道：" in extra:
        return True
    return False


def parse_heading(text: str, line_index: int) -> Optional[Heading]:
    if not text or len(text) > 60:
        return None
    if text.startswith(DIALOGUE_PREFIX):
        return None

    numbered = NUMBERED_HEADING.match(text)
    if numbered:
        number = chinese_to_int(numbered.group(1))
        if number is None:
            return None
        extra = (numbered.group(3) or "").strip()
        if is_sentence_title(extra):
            return None
        unit = numbered.group(2)
        kind = {"章": "chapter", "回": "hui", "节": "section"}.get(unit, "block")
        return Heading(line_index, text, text, kind, unit, number)

    latin = LATIN_CHAPTER.match(text)
    if latin:
        number = chinese_to_int(latin.group(1))
        extra = (latin.group(2) or "").strip()
        if number is None or is_sentence_title(extra):
            return None
        return Heading(line_index, text, text, "chapter", "Chapter", number)

    special = SPECIAL_HEADING.match(text)
    if special:
        return Heading(line_index, text, text, "special", "", None)
    return None


def sequence_score(headings: Sequence[Heading]) -> float:
    numbers = [h.number for h in headings if h.number is not None]
    if len(numbers) < 2:
        return 0.0 if not numbers else 0.4
    increasing = 0
    jumps = 0
    for prev, curr in zip(numbers, numbers[1:]):
        if curr > prev:
            increasing += 1
            if curr - prev > 5:
                jumps += 1
        elif curr == prev:
            increasing += 0.2
        else:
            increasing -= 1
    density = increasing / (len(numbers) - 1)
    unique_ratio = len(set(numbers)) / len(numbers)
    jump_penalty = jumps / (len(numbers) - 1)
    start_bonus = 0.15 if numbers[0] in (0, 1) else 0.0
    return density * 0.7 + unique_ratio * 0.3 + start_bonus - jump_penalty * 0.4


def spacing_score(headings: Sequence[Heading], line_count: int) -> float:
    if len(headings) < 2 or line_count <= 0:
        return 0.0
    spans = [b.line_index - a.line_index for a, b in zip(headings, headings[1:])]
    too_close = sum(1 for span in spans if span < 8)
    too_far = sum(1 for span in spans if span > max(8000, line_count // 2))
    close_ratio = too_close / len(spans)
    return 1.0 - close_ratio * 0.8 - (0.2 if too_far else 0.0)


def pick_best_headings(candidates: Sequence[Heading], line_count: int) -> List[Heading]:
    if not candidates:
        return []

    groups: dict[tuple, List[Heading]] = {}
    for item in candidates:
        key = (item.kind, item.unit)
        groups.setdefault(key, []).append(item)

    ranked: List[Tuple[float, List[Heading]]] = []
    for (kind, unit), items in groups.items():
        items = sorted(items, key=lambda h: h.line_index)
        count = len(items)
        seq = sequence_score(items) if kind != "special" else 0.55
        space = spacing_score(items, line_count)
        unit_bonus = {"章": 0.25, "回": 0.2, "Chapter": 0.2, "节": 0.05}.get(unit, 0.0)
        kind_bonus = 0.08 if kind == "chapter" else 0.0
        score = count * (1.0 + seq + space + unit_bonus + kind_bonus)
        ranked.append((score, items))

    ranked.sort(key=lambda pair: (pair[0], len(pair[1])), reverse=True)
    best = ranked[0][1]

    # 把楔子/后记等无编号标题并入主序列，但避开误伤正文。
    specials = groups.get(("special", ""), [])
    merged = list(best)
    used_lines = {h.line_index for h in merged}
    for item in specials:
        if item.line_index in used_lines:
            continue
        if any(abs(item.line_index - h.line_index) < 3 for h in merged):
            continue
        merged.append(item)
    merged.sort(key=lambda h: h.line_index)
    return drop_weak_duplicates(merged)


def drop_weak_duplicates(headings: Sequence[Heading]) -> List[Heading]:
    result: List[Heading] = []
    for item in headings:
        if result and item.line_index - result[-1].line_index < 3:
            prev = result[-1]
            if prev.kind == "special" and item.kind != "special":
                result[-1] = item
            continue
        if (
            result
            and item.number is not None
            and result[-1].number is not None
            and item.number < result[-1].number
            and item.kind == result[-1].kind
        ):
            # 编号突然回退，多半是正文里提到了前面章节。
            continue
        result.append(item)
    return result


def detect_headings(lines: Sequence[str]) -> List[Heading]:
    candidates: List[Heading] = []
    for index, raw in enumerate(lines):
        text = normalize_line(raw)
        heading = parse_heading(text, index)
        if heading is not None:
            candidates.append(heading)
    return pick_best_headings(candidates, len(lines))


def sanitize_filename(name: str, limit: int = 60) -> str:
    cleaned = INVALID_FILENAME.sub("_", name)
    cleaned = cleaned.strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip("_")
    return cleaned or "未命名章节"


def chapter_filename(index: int, heading: Optional[Heading], width: int) -> str:
    serial = f"{index:0{width}d}"
    if heading is None:
        return f"{serial}_文前.txt"
    return f"{serial}_{sanitize_filename(heading.title)}.txt"


def slice_chapters(lines: Sequence[str], headings: Sequence[Heading]) -> List[Tuple[Optional[Heading], str]]:
    pieces: List[Tuple[Optional[Heading], str]] = []
    starts = [h.line_index for h in headings]
    if not starts:
        body = "\n".join(lines).strip("\n")
        if body.strip():
            pieces.append((None, body + "\n"))
        return pieces

    preface = "\n".join(lines[: starts[0]]).strip("\n")
    if preface.strip():
        pieces.append((None, preface + "\n"))

    for idx, heading in enumerate(headings):
        begin = heading.line_index
        end = headings[idx + 1].line_index if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[begin:end]).strip("\n")
        if body.strip():
            pieces.append((heading, body + "\n"))
    return pieces


def output_dir_for(source: Path, out_root: Optional[Path]) -> Path:
    parent = out_root if out_root is not None else source.parent
    return parent / source.stem


def write_chapters(
    dest_dir: Path,
    pieces: Sequence[Tuple[Optional[Heading], str]],
    force: bool,
) -> List[Path]:
    if dest_dir.exists():
        if dest_dir.is_file():
            raise NotADirectoryError(f"输出路径已存在且不是文件夹: {dest_dir}")
        existing = list(dest_dir.iterdir())
        if existing and not force:
            raise FileExistsError(f"文件夹已存在且非空，如需覆盖请加 --force: {dest_dir}")
        if force:
            for old in dest_dir.glob("*.txt"):
                old.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(pieces))))
    has_preface = bool(pieces) and pieces[0][0] is None
    written: List[Path] = []
    for offset, (heading, body) in enumerate(pieces):
        serial = offset if has_preface else offset + 1
        path = dest_dir / chapter_filename(serial, heading, width)
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def build_plan(source: Path) -> SplitPlan:
    text, encoding = read_text(source)
    lines = text.splitlines()
    headings = detect_headings(lines)
    specials = [h for h in headings if h.kind == "special"]
    return SplitPlan(source, encoding, lines, headings, specials)


def preview(plan: SplitPlan) -> None:
    pieces = slice_chapters(plan.lines, plan.headings)
    print(f"文件: {plan.source}")
    print(f"编码: {plan.encoding}")
    print(f"总行数: {len(plan.lines)}")
    print(f"识别章节: {sum(1 for h, _ in pieces if h is not None)} 个")
    if pieces and pieces[0][0] is None:
        print(f"文前: {len(pieces[0][1])} 字")
    for heading, body in pieces:
        if heading is None:
            continue
        print(f"  L{heading.line_index + 1:>6}  {heading.title}  ({len(body)} 字)")


def process_one(source: Path, out_root: Optional[Path], dry_run: bool, force: bool) -> int:
    if not source.is_file():
        print(f"跳过（不是文件）: {source}", file=sys.stderr)
        return 1
    plan = build_plan(source)
    if dry_run:
        preview(plan)
        print()
        return 0 if plan.headings else 2

    pieces = slice_chapters(plan.lines, plan.headings)
    if not plan.headings:
        print(f"未识别到章节标题: {source}", file=sys.stderr)
        return 2
    dest = output_dir_for(source, out_root)
    written = write_chapters(dest, pieces, force=force)
    print(f"已拆分 {source.name} -> {dest}")
    print(f"  编码 {plan.encoding}，写出 {len(written)} 个文件")
    return 0


def iter_inputs(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.txt") if p.is_file()))
        else:
            files.append(path)
    # 避免把本脚本或已拆出的章节再拆一遍
    unique: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or path.name == Path(__file__).name:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能拆分小说 TXT，一章一个文件。")
    parser.add_argument("inputs", nargs="+", help="小说 txt 文件或包含 txt 的文件夹")
    parser.add_argument("-o", "--output", type=Path, help="输出根目录，默认与源文件同级")
    parser.add_argument("--dry-run", action="store_true", help="只预览识别结果，不写文件")
    parser.add_argument("--force", action="store_true", help="允许写入已存在的非空文件夹")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sources = iter_inputs(Path(item) for item in args.inputs)
    if not sources:
        print("没有找到可处理的 txt 文件", file=sys.stderr)
        return 1
    status = 0
    for source in sources:
        code = process_one(source, args.output, args.dry_run, args.force)
        status = max(status, code)
    return status


if __name__ == "__main__":
    sys.exit(main())
