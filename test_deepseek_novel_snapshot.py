#!/usr/bin/env python3
"""命令行：读取 txt 小说，调用接口生成 Anima / 自然语言 / Krea2 节点快照提示词。

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
        description="读取 txt 小说，调用接口生成 Anima / 自然语言 / Krea2 节点快照提示词。"
    )
    parser.add_argument(
        "novel",
        nargs="?",
        help="小说 txt 路径。省略则使用 samples/snapshot_test_excerpt.txt",
    )
    parser.add_argument(
        "--style",
        default=os.environ.get("DEEPSEEK_PROMPT_STYLE", core.DEFAULT_PROMPT_STYLE),
        help="提示词类型：danbooru、natural（自然语言）或 krea2。也可用环境变量 DEEPSEEK_PROMPT_STYLE",
    )
    parser.add_argument(
        "--nodes",
        default=os.environ.get("DEEPSEEK_NODE_COUNT", str(core.DEFAULT_NODE_COUNT)),
        help="固定分镜数，1 到 100。也可用环境变量 DEEPSEEK_NODE_COUNT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        prompt_style = core.normalize_prompt_style(args.style)
        node_count = core.normalize_node_count(args.nodes)
        novel_path = core.resolve_novel_path(args.novel)
        system_prompt = core.render_system_prompt(
            core.load_text(core.prompt_path_for(prompt_style)),
            node_count,
        )
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
    print(f"提示词类型: {core.PROMPT_STYLE_LABELS[prompt_style]} ({prompt_style})")
    print(f"分镜数: {node_count}")
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
            prompt_style=prompt_style,
            node_count=node_count,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n".join(result["report"]))
    print(f"完整输出已写入: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
