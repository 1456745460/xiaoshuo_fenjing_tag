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
        self.minsize(680, 580)
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
        self.node_count_var = tk.StringVar(value=str(saved_count))
        self.show_key = False
        self._build()
        self._center_window(760, 700)
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

        ttk.Label(form, text="模型").grid(row=3, column=0, sticky=tk.W, pady=6)
        model_row = ttk.Frame(form)
        model_row.grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=6)
        model_row.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var)
        saved_models = self.cfg.get("models") or []
        if isinstance(saved_models, list) and saved_models:
            self.model_combo["values"] = saved_models
        self.model_combo.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(model_row, text="获取模型", command=self.fetch_models).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(form, text="小说 txt").grid(row=4, column=0, sticky=tk.W, pady=6)
        file_row = ttk.Frame(form)
        file_row.grid(row=4, column=1, columnspan=2, sticky=tk.EW, pady=6)
        file_row.columnconfigure(0, weight=1)
        self.novel_path = ttk.Entry(file_row)
        self.novel_path.insert(0, str(self.cfg.get("novel_path") or ""))
        self.novel_path.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(file_row, text="选择文件", command=self.choose_txt).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(form, text="提示词类型").grid(row=5, column=0, sticky=tk.W, pady=6)
        style_row = ttk.Frame(form)
        style_row.grid(row=5, column=1, columnspan=2, sticky=tk.W, pady=6)
        ttk.Radiobutton(
            style_row,
            text="Danbooru",
            variable=self.prompt_style_var,
            value=core.PROMPT_STYLE_DANBOORU,
            command=self._persist,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            style_row,
            text="自然语言",
            variable=self.prompt_style_var,
            value=core.PROMPT_STYLE_NATURAL,
            command=self._persist,
        ).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Label(form, text="分镜数").grid(row=6, column=0, sticky=tk.W, pady=6)
        self.node_count_combo = ttk.Combobox(
            form,
            textvariable=self.node_count_var,
            values=core.node_count_choices(),
            state="readonly",
            width=8,
        )
        self.node_count_combo.grid(row=6, column=1, sticky=tk.W, pady=6)
        self.node_count_combo.bind("<<ComboboxSelected>>", lambda _event: self._persist())

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(16, 8))
        self.start_btn = ttk.Button(actions, text="开始生成", command=self.start_generate)
        self.start_btn.pack(side=tk.LEFT)
        self.open_dir_btn = ttk.Button(
            actions,
            text="打开文件所在目录",
            command=self.open_output_dir,
            state=tk.DISABLED,
        )
        self.open_dir_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status = ttk.Label(actions, text="就绪")
        self.status.pack(side=tk.LEFT, padx=12)

        ttk.Label(root, text="日志").pack(anchor=tk.W, pady=(8, 4))
        log_frame = ttk.Frame(root)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(log_frame, height=14, wrap=tk.WORD)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.insert(
            tk.END,
            "1. 填写 API 地址和 Key\n"
            "2. 点「获取模型」后选择模型\n"
            "3. 选择提示词类型：Danbooru 或 自然语言\n"
            "4. 下拉选择分镜数（10 到 20，生成固定数量）\n"
            "5. 选择 txt 小说，点「开始生成」\n"
            "6. 完成后会自动打开生成的 md，也可点「打开文件所在目录」\n",
        )
        self.log.configure(state=tk.DISABLED)

        for child in root.winfo_children():
            if isinstance(child, ttk.Frame):
                child.configure()
        _ = pad

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

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.start_btn.configure(state=state)
        self.status.configure(text=text)

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
            "model": self.model_var.get().strip(),
            "novel_path": self.novel_path.get().strip(),
            "prompt_style": self.prompt_style_var.get().strip(),
            "node_count": self.node_count_var.get().strip(),
            "models": list(self.model_combo["values"] or []),
        }

    def _persist(self) -> None:
        save_config(self._snapshot_fields())

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

    def start_generate(self) -> None:
        if self.busy:
            return
        fields = self._snapshot_fields()
        if not fields["api_url"] or not fields["api_key"]:
            messagebox.showwarning("缺少参数", "请先填写 API 地址和 API Key")
            return
        if not fields["model"]:
            messagebox.showwarning("缺少参数", "请先获取并选择模型")
            return
        if not fields["novel_path"]:
            messagebox.showwarning("缺少参数", "请选择 txt 小说文件")
            return
        try:
            max_tokens = int(fields["max_tokens"])
            if max_tokens <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("参数错误", "max_tokens 必须是正整数")
            return
        try:
            novel_path = core.resolve_novel_path(fields["novel_path"])
        except (FileNotFoundError, IsADirectoryError, OSError) as exc:
            messagebox.showerror("文件错误", str(exc))
            return
        try:
            prompt_style = core.normalize_prompt_style(fields["prompt_style"])
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return
        try:
            node_count = core.normalize_node_count(fields["node_count"])
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return

        self._persist()
        self._set_busy(True, "正在生成，可能需要 1~3 分钟...")
        style_label = core.PROMPT_STYLE_LABELS[prompt_style]
        self._append_log(
            f"开始生成：{novel_path.name} / {fields['model']} / "
            f"max_tokens={max_tokens} / 提示词={style_label} / 分镜数={node_count}"
        )

        def worker() -> None:
            try:
                result = core.generate_snapshot(
                    novel_path=novel_path,
                    api_url=fields["api_url"],
                    api_key=fields["api_key"],
                    model=fields["model"],
                    max_tokens=max_tokens,
                    prompt_style=prompt_style,
                    node_count=node_count,
                )
                self.after(0, lambda result=result: self._on_generated(result, None))
            except Exception as exc:
                self.after(0, lambda err=str(exc): self._on_generated(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generated(self, result: dict | None, error: str | None) -> None:
        self._set_busy(False, "就绪")
        if error or not result:
            self._append_log(error or "生成失败")
            messagebox.showerror("生成失败", error or "未知错误")
            return
        output_path: Path = result["output_path"]
        self.last_output_path = output_path
        self.open_dir_btn.configure(state=tk.NORMAL)
        self._append_log("\n".join(result["report"]))
        self._append_log(f"已写入: {output_path}")
        if result.get("truncated"):
            self._append_log("正文过长，已截取开头送入。")
        try:
            core.open_path(output_path)
            self.status.configure(text="生成完成，已打开 md")
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
