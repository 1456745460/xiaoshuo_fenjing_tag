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
    parser.add_argument(
        "--step",
        choices=("all", "3"),
        default="all",
        help="all=完整流水线；3=只用人物一致性和故事概括重跑 TAG（仅自然语言）",
    )
    parser.add_argument(
        "--character",
        help="人物一致性稿路径。省略则读取最近一次 outputs/年月日时分秒/1_character.md",
    )
    parser.add_argument(
        "--summary",
        help="故事概括稿路径。省略则读取最近一次 outputs/年月日时分秒/2_summary.md",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get(
            "SNAPSHOT_API_BACKEND",
            os.environ.get("DEEPSEEK_API_BACKEND", core.DEFAULT_API_BACKEND),
        ),
        help="API 协议：auto、chat_completions 或 responses。也可用 SNAPSHOT_API_BACKEND",
    )
    parser.add_argument(
        "--sexy-clothing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="开启后不跟故事原装，大胆改成更性感暴露的服装。也可用 SNAPSHOT_SEXY_CLOTHING",
    )
    return parser.parse_args(argv)


def load_material(path: str | None, fallback: Path) -> str:
    target = core.resolve_novel_path(path) if path else fallback
    text = core.load_text(target).strip()
    if not text:
        raise ValueError(f"材料文件是空的: {target}")
    print(f"读取材料: {target}（{len(text)} 字）")
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        prompt_style = core.normalize_prompt_style(args.style)
        node_count = core.normalize_node_count(args.nodes)
        novel_path = core.resolve_novel_path(args.novel)
        if args.sexy_clothing is None:
            sexy_clothing = core.normalize_sexy_clothing(
                os.environ.get("SNAPSHOT_SEXY_CLOTHING")
            )
        else:
            sexy_clothing = bool(args.sexy_clothing)
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.step == "3" and prompt_style != core.PROMPT_STYLE_NATURAL:
        print("只跑第三步仅支持 --style natural", file=sys.stderr)
        return 2

    api_key = (
        os.environ.get("SNAPSHOT_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GROK_API_KEY")
        or ""
    ).strip()
    if not api_key:
        print("缺少环境变量 SNAPSHOT_API_KEY / DEEPSEEK_API_KEY / GROK_API_KEY", file=sys.stderr)
        return 2

    api_url = os.environ.get(
        "SNAPSHOT_API_URL",
        os.environ.get("DEEPSEEK_API_URL", core.DEFAULT_API_URL),
    )
    model = os.environ.get(
        "SNAPSHOT_MODEL",
        os.environ.get("DEEPSEEK_MODEL", core.DEFAULT_MODEL),
    )
    max_tokens = int(
        os.environ.get(
            "SNAPSHOT_MAX_TOKENS",
            os.environ.get("DEEPSEEK_MAX_TOKENS", str(core.DEFAULT_MAX_TOKENS)),
        )
    )
    try:
        api_backend = core.normalize_api_backend(args.backend, model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"模型: {model}")
    print(f"协议: {core.API_BACKEND_LABELS.get(api_backend, api_backend)}")
    print(f"小说: {novel_path}")
    print(f"提示词类型: {core.PROMPT_STYLE_LABELS[prompt_style]} ({prompt_style})")
    print(f"分镜数: {node_count}")
    print(f"性感服装: {'开' if sexy_clothing else '关'}")

    try:
        if args.step == "3":
            character_path, summary_path = core.nl_draft_paths(novel_path)
            character_bible = load_material(args.character, character_path)
            summary = load_material(args.summary, summary_path)
            tag_prompt = core.render_system_prompt(
                core.load_text(core.prompt_path_for(prompt_style)),
                node_count,
                sexy_clothing=sexy_clothing,
            )
            print("流水线: 只跑第三步（按当前人物一致性与概括出 TAG）")
            print(f"第三步 TAG 提示词字数: {len(tag_prompt)}")
            print("正在调用接口...")
            result = core.generate_nl_tag_from_materials(
                novel_path=novel_path,
                character_bible=character_bible,
                summary=summary,
                api_url=api_url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                node_count=node_count,
                on_progress=print,
                api_backend=args.backend,
                sexy_clothing=sexy_clothing,
            )
        else:
            max_chars = int(
                os.environ.get("DEEPSEEK_NOVEL_MAX_CHARS", str(core.DEFAULT_NOVEL_MAX_CHARS))
            )
            novel, truncated = core.clip_novel(core.load_text(novel_path), max_chars)
            if not novel:
                print(f"小说文件是空的: {novel_path}", file=sys.stderr)
                return 2
            if prompt_style == core.PROMPT_STYLE_NATURAL:
                character_prompt = core.render_system_prompt(
                    core.load_text(core.NL_CHARACTER_PROMPT_PATH),
                    sexy_clothing=sexy_clothing,
                )
                summary_prompt = core.render_system_prompt(
                    core.load_text(core.NL_SUMMARY_PROMPT_PATH),
                    sexy_clothing=sexy_clothing,
                )
                tag_prompt = core.render_system_prompt(
                    core.load_text(core.prompt_path_for(prompt_style)),
                    node_count,
                    sexy_clothing=sexy_clothing,
                )
                print("流水线: 自然语言三步独立对话")
                print(f"第一步人物一致性提示词字数: {len(character_prompt)}")
                print(f"第二步小说概括提示词字数: {len(summary_prompt)}")
                print(f"第三步 TAG 提示词字数: {len(tag_prompt)}")
            else:
                system_prompt = core.render_system_prompt(
                    core.load_text(core.prompt_path_for(prompt_style)),
                    node_count,
                    sexy_clothing=sexy_clothing,
                )
                print(f"系统提示词字数: {len(system_prompt)}")
            print(f"小说送入字数: {len(novel)}")
            if truncated:
                print(
                    f"正文超过 {max_chars} 字，已截取开头送入。"
                    "可用 DEEPSEEK_NOVEL_MAX_CHARS 调整。"
                )
            print("正在调用接口...")
            result = core.generate_snapshot(
                novel_path=novel_path,
                api_url=api_url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                novel_max_chars=max_chars,
                prompt_style=prompt_style,
                node_count=node_count,
                on_progress=print,
                api_backend=args.backend,
                sexy_clothing=sexy_clothing,
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n".join(result["report"]))
    if result.get("run_dir"):
        print(f"输出目录: {result['run_dir']}")
    print(f"完整输出已写入: {result['output_path']}")
    if result.get("character_path"):
        print(f"人物一致性稿: {result['character_path']}")
    if result.get("summary_path"):
        print(f"故事概括稿: {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
