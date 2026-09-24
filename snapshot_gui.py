#!/usr/bin/env python3
"""小说节点快照可视化工具。"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import anima_snapshot as core

CONFIG_PATH = core.ROOT / "gui_config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class SnapshotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title("小说节点快照生成器")
        self.minsize(780, 680)
        self.cfg = load_config()
        self.busy = False
        self.last_output_path: Path | None = None
        self.model_var = tk.StringVar(value=str(self.cfg.get("model") or core.DEFAULT_MODEL))
        saved_style = str(self.cfg.get("prompt_style") or core.DEFAULT_PROMPT_STYLE)
        try:
            saved_style = core.normalize_prompt_style(saved_style)
        except ValueError:
            saved_style = core.DEFAULT_PROMPT_STYLE
        self.prompt_style_var = tk.StringVar(value=saved_style)
        try:
            saved_count = core.normalize_node_count(
                self.cfg.get("node_count") or core.DEFAULT_NODE_COUNT
            )
        except ValueError:
            saved_count = core.DEFAULT_NODE_COUNT
        self._saved_node_count = str(saved_count)
        try:
            saved_sexy = core.normalize_sexy_clothing(self.cfg.get("sexy_clothing"))
        except ValueError:
            saved_sexy = core.DEFAULT_SEXY_CLOTHING
        self.sexy_clothing_var = tk.BooleanVar(value=saved_sexy)
        self.show_key = False
        self._build()
        self._center_window(920, 820)
        self._try_load_drafts(silent=True)
        self._sync_nl_controls()
        self.deiconify()
        self.lift()
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        pad = {"padx": 16, "pady": 6}
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="小说节点快照生成器", font=("PingFang SC", 18, "bold")).pack(
            anchor=tk.W, pady=(0, 12)
        )

        form = ttk.Frame(root)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)

        self.api_url = self._add_entry(
            form, 0, "API 地址", self.cfg.get("api_url") or core.DEFAULT_API_URL
        )
        self.api_key = self._add_entry(
            form, 1, "API Key", self.cfg.get("api_key") or "", show="*"
        )
        ttk.Button(form, text="显示", width=8, command=self._toggle_key).grid(
            row=1, column=2, padx=(8, 0)
        )
        self.max_tokens = self._add_entry(
            form, 2, "max_tokens", str(self.cfg.get("max_tokens") or core.DEFAULT_MAX_TOKENS)
        )

        ttk.Label(form, text="API 协议").grid(row=3, column=0, sticky=tk.W, pady=6)
        self._backend_labels = {value: label for value, label in core.API_BACKEND_CHOICES}
        self._backend_values = {label: value for value, label in core.API_BACKEND_CHOICES}
        saved_backend = str(self.cfg.get("api_backend") or core.DEFAULT_API_BACKEND)
        if saved_backend not in self._backend_labels:
            saved_backend = core.DEFAULT_API_BACKEND
        self.backend_var = tk.StringVar(value=self._backend_labels[saved_backend])
        backend_row = ttk.Frame(form)
        backend_row.grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=6)
        backend_row.columnconfigure(0, weight=1)
        self.backend_combo = ttk.Combobox(
            backend_row,
            textvariable=self.backend_var,
            state="readonly",
            values=[choice[1] for choice in core.API_BACKEND_CHOICES],
        )
        self.backend_combo.grid(row=0, column=0, sticky=tk.EW)
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _event: self._persist())
        ttk.Label(
            backend_row,
            text="Grok 4.6 必须用 Responses",
        ).grid(row=0, column=1, padx=(8, 0), sticky=tk.W)

        ttk.Label(form, text="模型").grid(row=4, column=0, sticky=tk.W, pady=6)
        model_row = ttk.Frame(form)
        model_row.grid(row=4, column=1, columnspan=2, sticky=tk.EW, pady=6)
        model_row.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var)
        saved_models = self.cfg.get("models") or []
        if isinstance(saved_models, list) and saved_models:
            self.model_combo["values"] = saved_models
        self.model_combo.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(model_row, text="获取模型", command=self.fetch_models).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(form, text="小说 txt").grid(row=5, column=0, sticky=tk.W, pady=6)
        file_row = ttk.Frame(form)
        file_row.grid(row=5, column=1, columnspan=2, sticky=tk.EW, pady=6)
        file_row.columnconfigure(0, weight=1)
        self.novel_path = ttk.Entry(file_row)
        self.novel_path.insert(0, str(self.cfg.get("novel_path") or ""))
        self.novel_path.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(file_row, text="选择文件", command=self.choose_txt).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(form, text="提示词类型").grid(row=6, column=0, sticky=tk.W, pady=6)
        style_row = ttk.Frame(form)
        style_row.grid(row=6, column=1, columnspan=2, sticky=tk.W, pady=6)
        ttk.Radiobutton(
            style_row,
            text="Danbooru",
            variable=self.prompt_style_var,
            value=core.PROMPT_STYLE_DANBOORU,
            command=self._on_style_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            style_row,
            text="自然语言",
            variable=self.prompt_style_var,
            value=core.PROMPT_STYLE_NATURAL,
            command=self._on_style_change,
        ).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Radiobutton(
            style_row,
            text="Krea2",
            variable=self.prompt_style_var,
            value=core.PROMPT_STYLE_KREA2,
            command=self._on_style_change,
        ).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Label(form, text="分镜数").grid(row=7, column=0, sticky=tk.W, pady=6)
        count_row = ttk.Frame(form)
        count_row.grid(row=7, column=1, columnspan=2, sticky=tk.W, pady=6)
        self.node_count = ttk.Entry(count_row, width=8)
        self.node_count.insert(0, self._saved_node_count)
        self.node_count.pack(side=tk.LEFT)
        ttk.Label(count_row, text=f"正整数，{core.MIN_NODE_COUNT} 到 {core.MAX_NODE_COUNT}").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.node_count.bind("<FocusOut>", lambda _event: self._persist())
        self.node_count.bind("<Return>", lambda _event: self._persist())

        ttk.Label(form, text="性感服装").grid(row=8, column=0, sticky=tk.W, pady=6)
        sexy_row = ttk.Frame(form)
        sexy_row.grid(row=8, column=1, columnspan=2, sticky=tk.W, pady=6)
        ttk.Checkbutton(
            sexy_row,
            text="开启后不跟故事原装，大胆改成更暴露的服装",
            variable=self.sexy_clothing_var,
            command=self._persist,
        ).pack(side=tk.LEFT)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(16, 8))
        self.start_btn = ttk.Button(actions, text="开始生成", command=self.start_generate)
        self.start_btn.pack(side=tk.LEFT)
        self.step3_btn = ttk.Button(
            actions,
            text="只跑第三步出 TAG",
            command=self.start_step3,
        )
        self.step3_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.save_drafts_btn = ttk.Button(
            actions,
            text="保存中间稿",
            command=self.save_drafts,
        )
        self.save_drafts_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.open_dir_btn = ttk.Button(
            actions,
            text="打开文件所在目录",
            command=self.open_output_dir,
            state=tk.DISABLED,
        )
        self.open_dir_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status = ttk.Label(actions, text="就绪")
        self.status.pack(side=tk.LEFT, padx=12)

        ttk.Label(
            root,
            text="自然语言：前两步结果会出现在「人物一致性」「故事概括」页，改完后可只重跑第三步。勾选「性感服装」后，出图服装不跟故事原装。",
        ).pack(anchor=tk.W, pady=(0, 4))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        log_tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(log_tab, text="日志")
        self.log = tk.Text(log_tab, height=14, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_tab, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.insert(
            tk.END,
            "1. 填写 API 地址和 Key。Grok / packyapi 填 https://www.packyapi.ai/v1\n"
            "2. API 协议选「自动」即可：grok-4.6 走 Responses，DeepSeek 走 Chat Completions\n"
            "3. 点「获取模型」后选择模型\n"
            "4. 选择提示词类型：Danbooru（Anima tag）、自然语言、或 Krea2（二次元分镜简报，贴英文）\n"
            "   自然语言会分三次独立对话：人物一致性 → 详细概括 → Anima 混合 TAG（外形词标签+短句，不要小说修辞）\n"
            f"5. 填写分镜数（{core.MIN_NODE_COUNT} 到 {core.MAX_NODE_COUNT}，生成固定数量）\n"
            "6. 需要更暴露的服装时勾选「性感服装」：不跟故事原装，大胆改品类/颜色/暴露/花纹/样式/点缀\n"
            "7. 选择 txt 小说，点「开始生成」\n"
            "8. 自然语言生成后，可在「人物一致性」「故事概括」页直接改稿，再点「只跑第三步出 TAG」\n"
            "9. 完成后会自动打开生成的 md，也可点「打开文件所在目录」\n"
            "   每次生成会新建 outputs/年月日时分秒/，自然语言写入 1_character.md、2_summary.md、3_tags.md\n",
        )
        self.log.configure(state=tk.DISABLED)

        self.character_text = self._make_editor_tab("人物一致性")
        self.summary_text = self._make_editor_tab("故事概括")

        for child in root.winfo_children():
            if isinstance(child, ttk.Frame):
                child.configure()
        _ = pad

    def _make_editor_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(frame, text=title)
        text = tk.Text(frame, wrap=tk.WORD, undo=True)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.bind("<KeyRelease>", lambda _event: self._sync_nl_controls())
        return text

    def _center_window(self, width: int, height: int) -> None:
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _add_entry(self, parent: ttk.Frame, row: int, label: str, value: str, show: str = "") -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=6)
        entry = ttk.Entry(parent, show=show)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky=tk.EW, pady=6)
        return entry

    def _toggle_key(self) -> None:
        self.show_key = not self.show_key
        self.api_key.configure(show="" if self.show_key else "*")

    def _nl_mode(self) -> bool:
        try:
            return core.normalize_prompt_style(self.prompt_style_var.get()) == core.PROMPT_STYLE_NATURAL
        except ValueError:
            return False

    def _editor_text(self, widget: tk.Text) -> str:
        return widget.get("1.0", tk.END).strip()

    def _set_editor(self, widget: tk.Text, content: str) -> None:
        widget.delete("1.0", tk.END)
        if content.strip():
            widget.insert("1.0", content.strip() + "\n")

    def _sync_nl_controls(self) -> None:
        if not hasattr(self, "step3_btn"):
            return
        nl = self._nl_mode()
        idle = not self.busy
        has_drafts = bool(self._editor_text(self.character_text)) and bool(
            self._editor_text(self.summary_text)
        )
        self.step3_btn.configure(state=tk.NORMAL if nl and idle and has_drafts else tk.DISABLED)
        self.save_drafts_btn.configure(state=tk.NORMAL if nl and idle else tk.DISABLED)

    def _on_style_change(self) -> None:
        self._persist()
        self._sync_nl_controls()

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.start_btn.configure(state=state)
        self.status.configure(text=text)
        self._sync_nl_controls()

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _snapshot_fields(self) -> dict:
        return {
            "api_url": self.api_url.get().strip(),
            "api_key": self.api_key.get().strip(),
            "max_tokens": self.max_tokens.get().strip(),
            "api_backend": self._backend_values.get(
                self.backend_var.get().strip(),
                core.DEFAULT_API_BACKEND,
            ),
            "model": self.model_var.get().strip(),
            "novel_path": self.novel_path.get().strip(),
            "prompt_style": self.prompt_style_var.get().strip(),
            "node_count": self.node_count.get().strip(),
            "sexy_clothing": bool(self.sexy_clothing_var.get()),
            "models": list(self.model_combo["values"] or []),
        }

    def _persist(self) -> None:
        save_config(self._snapshot_fields())

    def _try_load_drafts(self, silent: bool = False) -> None:
        raw = self.novel_path.get().strip()
        if not raw:
            return
        try:
            novel_path = core.resolve_novel_path(raw)
            character, summary = core.load_nl_drafts(novel_path)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError):
            return
        self._set_editor(self.character_text, character)
        self._set_editor(self.summary_text, summary)
        self._sync_nl_controls()
        if not silent:
            character_path, summary_path = core.nl_draft_paths(novel_path)
            self._append_log(f"已载入中间稿：{character_path} ； {summary_path}")

    def choose_txt(self) -> None:
        current = self.novel_path.get().strip()
        initial = str(Path(current).parent) if current else str(core.ROOT)
        path = filedialog.askopenfilename(
            title="选择小说 txt",
            initialdir=initial,
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.novel_path.delete(0, tk.END)
        self.novel_path.insert(0, path)
        self._persist()
        self._try_load_drafts()

    def fetch_models(self) -> None:
        if self.busy:
            return
        fields = self._snapshot_fields()
        if not fields["api_url"] or not fields["api_key"]:
            messagebox.showwarning("缺少参数", "请先填写 API 地址和 API Key")
            return
        self._set_busy(True, "正在获取模型...")
        self._append_log("开始获取模型列表...")

        def worker() -> None:
            try:
                models = core.list_models(fields["api_url"], fields["api_key"])
                self.after(0, lambda models=models: self._on_models(models, None))
            except Exception as exc:
                self.after(0, lambda err=str(exc): self._on_models([], err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_models(self, models: list[str], error: str | None) -> None:
        self._set_busy(False, "就绪")
        if error:
            self._append_log(error)
            messagebox.showerror("获取模型失败", error)
            return
        self.model_combo["values"] = models
        current = self.model_var.get().strip()
        if current not in models:
            self.model_var.set(models[0])
        self._append_log("可用模型：\n- " + "\n- ".join(models))
        self._persist()

    def _common_generate_args(self) -> dict | None:
        fields = self._snapshot_fields()
        if not fields["api_url"] or not fields["api_key"]:
            messagebox.showwarning("缺少参数", "请先填写 API 地址和 API Key")
            return None
        if not fields["model"]:
            messagebox.showwarning("缺少参数", "请先获取并选择模型")
            return None
        if not fields["novel_path"]:
            messagebox.showwarning("缺少参数", "请选择 txt 小说文件")
            return None
        try:
            max_tokens = int(fields["max_tokens"])
            if max_tokens <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("参数错误", "max_tokens 必须是正整数")
            return None
        try:
            novel_path = core.resolve_novel_path(fields["novel_path"])
        except (FileNotFoundError, IsADirectoryError, OSError) as exc:
            messagebox.showerror("文件错误", str(exc))
            return None
        try:
            prompt_style = core.normalize_prompt_style(fields["prompt_style"])
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return None
        try:
            node_count = core.normalize_node_count(fields["node_count"])
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return None
        try:
            sexy_clothing = core.normalize_sexy_clothing(fields.get("sexy_clothing"))
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return None
        try:
            resolved_backend = core.normalize_api_backend(
                fields.get("api_backend"), fields["model"]
            )
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return None
        self._persist()
        return {
            "fields": fields,
            "max_tokens": max_tokens,
            "novel_path": novel_path,
            "prompt_style": prompt_style,
            "node_count": node_count,
            "sexy_clothing": sexy_clothing,
            "api_backend": fields.get("api_backend") or core.DEFAULT_API_BACKEND,
            "resolved_backend": resolved_backend,
        }

    def start_generate(self) -> None:
        if self.busy:
            return
        args = self._common_generate_args()
        if args is None:
            return
        fields = args["fields"]
        novel_path = args["novel_path"]
        prompt_style = args["prompt_style"]
        max_tokens = args["max_tokens"]
        node_count = args["node_count"]
        sexy_clothing = args["sexy_clothing"]
        nl_mode = prompt_style == core.PROMPT_STYLE_NATURAL
        wait_hint = (
            "正在生成，自然语言约 3~8 分钟（三次独立对话）..."
            if nl_mode
            else "正在生成，可能需要 1~3 分钟..."
        )
        self._set_busy(True, wait_hint)
        style_label = core.PROMPT_STYLE_LABELS[prompt_style]
        backend_label = core.API_BACKEND_LABELS.get(
            args["resolved_backend"], args["resolved_backend"]
        )
        self._append_log(
            f"开始生成：{novel_path.name} / {fields['model']} / "
            f"协议={backend_label} / max_tokens={max_tokens} / "
            f"提示词={style_label} / 分镜数={node_count} / "
            f"性感服装={'开' if sexy_clothing else '关'}"
        )
        if nl_mode:
            self._append_log(
                "自然语言流水线：三次全新对话"
                " → 1) 人物一致性  2) 详细概括（保留故事结构与不同姿势）  3) 出 TAG"
            )

        def worker() -> None:
            def on_progress(message: str) -> None:
                self.after(0, lambda text=message: self._on_progress(text))

            def on_step_result(kind: str, content: str) -> None:
                self.after(0, lambda k=kind, c=content: self._apply_step_result(k, c))

            try:
                result = core.generate_snapshot(
                    novel_path=novel_path,
                    api_url=fields["api_url"],
                    api_key=fields["api_key"],
                    model=fields["model"],
                    max_tokens=max_tokens,
                    prompt_style=prompt_style,
                    node_count=node_count,
                    on_progress=on_progress,
                    on_step_result=on_step_result,
                    api_backend=args["api_backend"],
                    sexy_clothing=sexy_clothing,
                )
                self.after(0, lambda result=result: self._on_generated(result, None))
            except Exception as exc:
                self.after(0, lambda err=str(exc): self._on_generated(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def save_drafts(self) -> tuple[Path, Path] | None:
        raw = self.novel_path.get().strip()
        if not raw:
            messagebox.showwarning("缺少参数", "请先选择 txt 小说文件")
            return None
        try:
            novel_path = core.resolve_novel_path(raw)
        except (FileNotFoundError, IsADirectoryError, OSError) as exc:
            messagebox.showerror("文件错误", str(exc))
            return None
        character = self._editor_text(self.character_text)
        summary = self._editor_text(self.summary_text)
        if not character or not summary:
            messagebox.showwarning("缺少内容", "人物一致性和故事概括都不能为空")
            return None
        paths = core.write_nl_drafts(novel_path, character, summary)
        self._append_log(f"已保存中间稿：{paths[0]} ； {paths[1]}")
        self._sync_nl_controls()
        return paths

    def start_step3(self) -> None:
        if self.busy:
            return
        if not self._nl_mode():
            messagebox.showwarning("提示词类型", "只跑第三步仅支持自然语言")
            return
        args = self._common_generate_args()
        if args is None:
            return
        character = self._editor_text(self.character_text)
        summary = self._editor_text(self.summary_text)
        if not character or not summary:
            messagebox.showwarning("缺少内容", "请先有人物一致性和故事概括，或先跑完整三步")
            return
        fields = args["fields"]
        novel_path = args["novel_path"]
        self._set_busy(True, "正在按当前稿件重跑第三步...")
        backend_label = core.API_BACKEND_LABELS.get(
            args["resolved_backend"], args["resolved_backend"]
        )
        self._append_log(
            f"只跑第三步：{novel_path.name} / {fields['model']} / "
            f"协议={backend_label} / 分镜数={args['node_count']} / "
            f"性感服装={'开' if args['sexy_clothing'] else '关'}"
        )
        self._append_log("将使用当前「人物一致性」和「故事概括」页的文本，不再读小说原文。")

        def worker() -> None:
            def on_progress(message: str) -> None:
                self.after(0, lambda text=message: self._on_progress(text))

            try:
                result = core.generate_nl_tag_from_materials(
                    novel_path=novel_path,
                    character_bible=character,
                    summary=summary,
                    api_url=fields["api_url"],
                    api_key=fields["api_key"],
                    model=fields["model"],
                    max_tokens=args["max_tokens"],
                    node_count=args["node_count"],
                    on_progress=on_progress,
                    api_backend=args["api_backend"],
                    sexy_clothing=args["sexy_clothing"],
                )
                self.after(0, lambda result=result: self._on_generated(result, None, step3=True))
            except Exception as exc:
                self.after(0, lambda err=str(exc): self._on_generated(None, err, step3=True))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, message: str) -> None:
        self._append_log(message)
        self.status.configure(text=message)

    def _apply_step_result(self, kind: str, content: str) -> None:
        if kind == "character":
            self._set_editor(self.character_text, content)
            self._append_log("第一步完成，已写入「人物一致性」页，可随时修改。")
        elif kind == "summary":
            self._set_editor(self.summary_text, content)
            self._append_log("第二步完成，已写入「故事概括」页，可随时修改。")
        self._sync_nl_controls()

    def _on_generated(
        self,
        result: dict | None,
        error: str | None,
        step3: bool = False,
    ) -> None:
        self._set_busy(False, "就绪")
        if error or not result:
            self._append_log(error or "生成失败")
            messagebox.showerror("生成失败", error or "未知错误")
            return
        output_path: Path = result["output_path"]
        self.last_output_path = output_path
        self.open_dir_btn.configure(state=tk.NORMAL)
        if result.get("character_bible"):
            self._set_editor(self.character_text, str(result["character_bible"]))
        if result.get("summary"):
            self._set_editor(self.summary_text, str(result["summary"]))
        self._sync_nl_controls()
        self._append_log("\n".join(result["report"]))
        if result.get("run_dir"):
            self._append_log(f"输出目录: {result['run_dir']}")
        self._append_log(f"已写入: {output_path}")
        if result.get("character_path"):
            self._append_log(f"人物一致性稿: {result['character_path']}")
        if result.get("summary_path"):
            self._append_log(f"故事概括稿: {result['summary_path']}")
        if result.get("truncated"):
            self._append_log("正文过长，已截取开头送入。")
        done_text = "第三步完成，已打开 md" if step3 else "生成完成，已打开 md"
        try:
            core.open_path(output_path)
            self.status.configure(text=done_text)
        except OSError as exc:
            self._append_log(f"文件已生成，但自动打开失败: {exc}")
            messagebox.showinfo("生成完成", f"已保存到:\n{output_path}")

    def open_output_dir(self) -> None:
        if self.last_output_path is None:
            messagebox.showinfo("还没有生成文件", "请先完成一次生成")
            return
        try:
            core.open_dir(self.last_output_path)
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))

    def _on_close(self) -> None:
        try:
            self._persist()
        except OSError:
            pass
        self.destroy()


def main() -> None:
    app = SnapshotApp()
    app.mainloop()


if __name__ == "__main__":
    main()
