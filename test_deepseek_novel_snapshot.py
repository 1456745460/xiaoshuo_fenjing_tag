#!/usr/bin/env python3
"""命令行：读取 txt 小说，调用接口生成 Anima 节点快照提示词。

用法:
  python3 test_deepseek_novel_snapshot.py
  python3 test_deepseek_novel_snapshot.py 小说.txt
  python3 snapshot_gui.py
"""

from __future__ import annotations

import argparse
import os
import sys

import anima_snapshot as core


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 txt 小说，调用接口生成 Anima 节点快照提示词。"
    )
    parser.add_argument(
        "novel",
        nargs="?",
        help="小说 txt 路径。省略则使用 samples/snapshot_test_excerpt.txt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        novel_path = core.resolve_novel_path(args.novel)
        system_prompt = core.load_text(core.PROMPT_PATH)
        max_chars = int(os.environ.get("DEEPSEEK_NOVEL_MAX_CHARS", str(core.DEFAULT_NOVEL_MAX_CHARS)))
        novel, truncated = core.clip_novel(core.load_text(novel_path), max_chars)
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not novel:
        print(f"小说文件是空的: {novel_path}", file=sys.stderr)
        return 2

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("缺少环境变量 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    api_url = os.environ.get("DEEPSEEK_API_URL", core.DEFAULT_API_URL)
    model = os.environ.get("DEEPSEEK_MODEL", core.DEFAULT_MODEL)
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", str(core.DEFAULT_MAX_TOKENS)))

    print(f"模型: {model}")
    print(f"小说: {novel_path}")
    print(f"系统提示词字数: {len(system_prompt)}")
    print(f"小说送入字数: {len(novel)}")
    if truncated:
        print(f"正文超过 {max_chars} 字，已截取开头送入。可用 DEEPSEEK_NOVEL_MAX_CHARS 调整。")
    print("正在调用接口...")

    try:
        result = core.generate_snapshot(
            novel_path=novel_path,
            api_url=api_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            novel_max_chars=max_chars,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n".join(result["report"]))
    print(f"完整输出已写入: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
