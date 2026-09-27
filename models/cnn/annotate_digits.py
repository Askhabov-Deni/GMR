"""
annotate_digits.py — инструмент разметки кропов цифр для CNN.

Запуск:
    python annotate_digits.py [--crops_dir ./crops] [--model ./digit_cnn.pt]

Структура output  (<crops_dir>/labeled/):
    0/  1/  2/  ...  9/  trash/

Прогресс сохраняется в <crops_dir>/labeled/progress.json автоматически.

Горячие клавиши:
    0-9        — назначить метку и перейти к следующему
    T / Delete — переместить в trash/
    Space      — пропустить (без метки)
    Z          — undo
    Q / Escape — выйти (прогресс сохраняется)
"""

import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

# ─────────────────────────── опциональный torch ───────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ──────────────────────────── загрузка модели ─────────────────────────────────

def load_model(model_path: str):
    """TorchScript → state_dict fallback. Возвращает (model|None, transform|None)."""
    if not TORCH_OK or not model_path or not Path(model_path).exists():
        return None, None

    device = torch.device("cpu")

    # TorchScript
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"[model] TorchScript: {model_path}")
        return model, _make_transform()
    except Exception:
        pass

    # state_dict + DigitCNN
    try:
        script_dir = str(Path(__file__).parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from model_cnn import DigitCNN
        m = DigitCNN()
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state)
        m.eval()
        print(f"[model] state_dict: {model_path}")
        return m, _make_transform()
    except Exception as e:
        print(f"[model] не удалось загрузить: {e}")
        return None, None


def _make_transform(crop_h: int = 64, crop_w: int = 32):
    try:
        import config_cnn
        crop_h, crop_w = config_cnn.CROP_H, config_cnn.CROP_W
    except Exception:
        pass
    return T.Compose([
        T.Resize((crop_h, crop_w)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def predict(model, transform, image_path: str, top_k: int = 3):
    if model is None or transform is None:
        return []
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(probs)))
        return [(int(i), float(p)) for i, p in zip(top.indices, top.values)]
    except Exception as e:
        print(f"[predict] {e}")
        return []


# ──────────────────────────── стартовый диалог ────────────────────────────────

class SetupDialog(tk.Toplevel):
    """Окно выбора папки с кропами и (опционально) модели перед стартом."""

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Digit Annotator — настройка")
        self.resizable(False, False)
        self.grab_set()

        BG, FG, ACCENT = "#1e1e2e", "#cdd6f4", "#89b4fa"
        self.configure(bg=BG)

        self.crops_dir: str = ""
        self.model_path: str = ""
        self._cancelled = True

        pad = {"padx": 14, "pady": 6}

        # ── папка с кропами ──
        tk.Label(self, text="Папка с кропами:", bg=BG, fg=FG,
                 font=("Consolas", 10)).grid(row=0, column=0, sticky="w", **pad)

        crops_row = tk.Frame(self, bg=BG)
        crops_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14)

        self._crops_var = tk.StringVar()
        tk.Entry(crops_row, textvariable=self._crops_var, width=42,
                 bg="#313244", fg=FG, insertbackground=FG,
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=(0, 6))
        tk.Button(crops_row, text="Обзор…", command=self._pick_crops,
                  bg="#45475a", fg=FG, activebackground=ACCENT,
                  relief="flat", font=("Consolas", 9), padx=8).pack(side="left")

        # ── модель ──
        tk.Label(self, text="Модель .pt  (необязательно):", bg=BG, fg="#6c7086",
                 font=("Consolas", 10)).grid(row=2, column=0, sticky="w", **pad)

        model_row = tk.Frame(self, bg=BG)
        model_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14)

        self._model_var = tk.StringVar()
        tk.Entry(model_row, textvariable=self._model_var, width=42,
                 bg="#313244", fg=FG, insertbackground=FG,
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=(0, 6))
        tk.Button(model_row, text="Обзор…", command=self._pick_model,
                  bg="#45475a", fg=FG, activebackground=ACCENT,
                  relief="flat", font=("Consolas", 9), padx=8).pack(side="left")

        # ── кнопки ──
        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=4, column=0, columnspan=2, pady=14)

        tk.Button(btn_row, text="Начать", command=self._ok,
                  bg=ACCENT, fg="#1e1e2e", activebackground="#b4d0f8",
                  relief="flat", font=("Consolas", 11, "bold"),
                  padx=20, pady=6).pack(side="left", padx=10)
        tk.Button(btn_row, text="Отмена", command=self.destroy,
                  bg="#313244", fg="#f38ba8", activebackground="#45475a",
                  relief="flat", font=("Consolas", 10),
                  padx=12, pady=6).pack(side="left", padx=10)

        self.wait_window(self)

    def _pick_crops(self):
        d = filedialog.askdirectory(title="Папка с кропами")
        if d:
            self._crops_var.set(d)

    def _pick_model(self):
        f = filedialog.askopenfilename(
            title="Выберите модель",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("Все файлы", "*.*")],
        )
        if f:
            self._model_var.set(f)

    def _ok(self):
        crops = self._crops_var.get().strip()
        if not crops or not Path(crops).is_dir():
            messagebox.showwarning("Ошибка", "Укажите корректную папку с кропами.")
            return
        self.crops_dir = crops
        self.model_path = self._model_var.get().strip()
        self._cancelled = False
        self.destroy()


# ─────────────────────────────── основное окно ────────────────────────────────

PROGRESS_FILE = "progress.json"


class AnnotatorApp:
    DISPLAY_SIZE = 256
    HIGH_CONF = 0.90

    def __init__(self, root: tk.Tk, crops_dir: str, model, transform):
        self.root = root
        self.root.title("Digit Annotator")
        self.root.resizable(False, False)

        self.crops_dir = Path(crops_dir)
        self.output_dir = self.crops_dir / "labeled"
        self.progress_path = self.output_dir / PROGRESS_FILE
        self.model = model
        self.transform = transform

        for cls in list(map(str, range(10))) + ["trash"]:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_files = sorted(p for p in self.crops_dir.iterdir()
                           if p.suffix.lower() in exts)

        # ── восстановление прогресса ──
        self.labeled: dict[str, str] = {}   # filename → label
        self.history: list[str] = []         # ordered list of labeled filenames (for undo)
        self._load_progress()

        # пропускаем уже размеченные
        self.files = [f for f in all_files if f.name not in self.labeled]
        self.idx = 0

        # счётчики — заполняем из сохранённой разметки
        self.stats = {str(i): 0 for i in range(10)}
        self.stats["trash"] = 0
        self.stats["skip"] = 0
        for lbl in self.labeled.values():
            if lbl in self.stats:
                self.stats[lbl] += 1

        self._build_ui()
        self._refresh_stats()
        self._show_current()
        self._bind_keys()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────── прогресс ────────────

    def _load_progress(self):
        if self.progress_path.exists():
            try:
                data = json.loads(self.progress_path.read_text(encoding="utf-8"))
                self.labeled = data.get("labeled", {})
                self.history = data.get("history", [])
                print(f"[progress] восстановлено {len(self.labeled)} записей")
            except Exception as e:
                print(f"[progress] не удалось загрузить: {e}")

    def _save_progress(self):
        try:
            data = {"labeled": self.labeled, "history": self.history}
            self.progress_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[progress] не удалось сохранить: {e}")

    # ──────────── UI ────────────

    def _build_ui(self):
        BG, FG, ACCENT = "#1e1e2e", "#cdd6f4", "#89b4fa"
        self.root.configure(bg=BG)

        # изображение
        img_frame = tk.Frame(self.root, bg=BG, padx=16, pady=12)
        img_frame.grid(row=0, column=0, columnspan=2)
        self.img_label = tk.Label(img_frame, bg="#313244", relief="flat",
                                  width=self.DISPLAY_SIZE, height=self.DISPLAY_SIZE)
        self.img_label.pack()

        # имя файла
        self.fname_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.fname_var,
                 bg=BG, fg="#6c7086", font=("Consolas", 9)).grid(
            row=1, column=0, columnspan=2, pady=(0, 4))

        # предсказания
        pred_frame = tk.Frame(self.root, bg=BG)
        pred_frame.grid(row=2, column=0, columnspan=2, pady=4)
        tk.Label(pred_frame, text="Модель:", bg=BG, fg="#6c7086",
                 font=("Consolas", 10)).pack(side="left", padx=(0, 8))
        self.pred_labels: list[tk.Label] = []
        for _ in range(3):
            lbl = tk.Label(pred_frame, text="—", bg="#45475a", fg=FG,
                           font=("Consolas", 13, "bold"),
                           width=7, relief="flat", padx=4, pady=2)
            lbl.pack(side="left", padx=3)
            self.pred_labels.append(lbl)

        # прогресс
        self.progress_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.progress_var,
                 bg=BG, fg=ACCENT, font=("Consolas", 10)).grid(
            row=3, column=0, columnspan=2, pady=4)

        # кнопки цифр
        btn_frame = tk.Frame(self.root, bg=BG, padx=12)
        btn_frame.grid(row=4, column=0, pady=6)
        for i in range(10):
            tk.Button(btn_frame, text=str(i), width=3, height=1,
                      font=("Consolas", 13, "bold"),
                      bg="#313244", fg=FG,
                      activebackground=ACCENT, activeforeground=BG,
                      relief="flat", cursor="hand2",
                      command=lambda d=i: self._assign(str(d))).grid(
                row=0, column=i, padx=2)

        # кнопки управления
        ctrl_frame = tk.Frame(self.root, bg=BG, padx=12)
        ctrl_frame.grid(row=5, column=0, pady=6)
        for text, cmd, color in [
            ("Trash [T]",   lambda: self._assign("trash"), "#f38ba8"),
            ("Skip [Space]", self._skip,                   "#a6e3a1"),
            ("Undo [Z]",     self._undo,                   "#fab387"),
        ]:
            tk.Button(ctrl_frame, text=text, command=cmd,
                      font=("Consolas", 10), bg="#313244", fg=color,
                      activebackground=color, activeforeground=BG,
                      relief="flat", cursor="hand2", padx=10, pady=4).pack(
                side="left", padx=6)

        # статистика
        stats_frame = tk.LabelFrame(self.root, text=" Размечено ", bg=BG, fg="#6c7086",
                                    font=("Consolas", 9), padx=10, pady=6)
        stats_frame.grid(row=6, column=0, columnspan=2, padx=16, pady=(4, 12), sticky="ew")
        self.stat_labels: dict[str, tk.Label] = {}
        for col, key in enumerate(list(map(str, range(10))) + ["trash", "skip"]):
            tk.Label(stats_frame, text=key, bg=BG, fg="#6c7086",
                     font=("Consolas", 8)).grid(row=0, column=col, padx=4)
            lbl = tk.Label(stats_frame, text="0", bg=BG, fg=FG,
                           font=("Consolas", 10, "bold"))
            lbl.grid(row=1, column=col, padx=4)
            self.stat_labels[key] = lbl

        # хинт
        tk.Label(self.root,
                 text="[0-9] метка  [T/Del] trash  [Space] пропустить  [Z] undo  [Q/Esc] выход",
                 bg=BG, fg="#585b70", font=("Consolas", 8)).grid(
            row=7, column=0, columnspan=2, pady=(0, 10))

    def _refresh_stats(self):
        for key, lbl in self.stat_labels.items():
            lbl.configure(text=str(self.stats.get(key, 0)))

    # ──────────── логика ────────────

    def _show_current(self):
        if self.idx >= len(self.files):
            self._done()
            return

        path = self.files[self.idx]
        self.fname_var.set(path.name)
        done = len(self.labeled)
        total = done + len(self.files)
        self.progress_var.set(
            f"{self.idx + 1} / {len(self.files)} осталось"
            f"  •  всего размечено: {done} / {total}"
        )

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.DISPLAY_SIZE, self.DISPLAY_SIZE), Image.NEAREST)
            self._tk_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=self._tk_img,
                                     width=self.DISPLAY_SIZE, height=self.DISPLAY_SIZE)
        except Exception as e:
            self.img_label.configure(image="", text=f"Ошибка: {e}")

        preds = predict(self.model, self.transform, str(path))
        for i, lbl in enumerate(self.pred_labels):
            if i < len(preds):
                cls, conf = preds[i]
                bg = "#a6e3a1" if (i == 0 and conf >= self.HIGH_CONF) else "#45475a"
                fg = "#1e1e2e" if bg != "#45475a" else "#cdd6f4"
                lbl.configure(text=f"{cls}  {conf:.0%}", bg=bg, fg=fg)
            else:
                lbl.configure(text="—", bg="#45475a", fg="#cdd6f4")

    def _assign(self, label: str):
        if self.idx >= len(self.files):
            return
        src = self.files[self.idx]
        dst = self.output_dir / label / src.name
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.labeled[src.name] = label
        self.history.append(src.name)
        self.stats[label] = self.stats.get(label, 0) + 1
        self.stat_labels[label].configure(text=str(self.stats[label]))
        self._save_progress()

        self.idx += 1
        self._show_current()

    def _skip(self):
        self.stats["skip"] = self.stats.get("skip", 0) + 1
        self.stat_labels["skip"].configure(text=str(self.stats["skip"]))
        self.idx += 1
        self._show_current()

    def _undo(self):
        if not self.history:
            return
        fname = self.history.pop()
        label = self.labeled.pop(fname, None)
        if label:
            dst = self.output_dir / label / fname
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            self.stats[label] = max(0, self.stats.get(label, 1) - 1)
            self.stat_labels[label].configure(text=str(self.stats[label]))

        # возвращаем файл в очередь
        src = self.crops_dir / fname
        if src.exists() and (not self.files or self.files[self.idx - 1].name != fname):
            self.files.insert(self.idx, src)
        self.idx = max(0, self.idx - 1)
        self._save_progress()
        self._show_current()

    def _done(self):
        self._save_progress()
        total = sum(v for k, v in self.stats.items() if k not in ("skip",))
        messagebox.showinfo(
            "Готово",
            f"Все кропы размечены!\n\n"
            f"Размечено: {total}\n"
            f"Пропущено: {self.stats['skip']}\n\n"
            f"Результаты: {self.output_dir}",
        )
        self.root.quit()

    def _on_close(self):
        self._save_progress()
        self.root.quit()

    def _bind_keys(self):
        for i in range(10):
            self.root.bind(str(i), lambda e, d=i: self._assign(str(d)))
        self.root.bind("<KP_0>", lambda e: self._assign("0"))
        for i in range(1, 10):
            self.root.bind(f"<KP_{i}>", lambda e, d=i: self._assign(str(d)))
        self.root.bind("t", lambda e: self._assign("trash"))
        self.root.bind("T", lambda e: self._assign("trash"))
        self.root.bind("<Delete>", lambda e: self._assign("trash"))
        self.root.bind("<space>", lambda e: self._skip())
        self.root.bind("z", lambda e: self._undo())
        self.root.bind("Z", lambda e: self._undo())
        self.root.bind("q", lambda e: self._on_close())
        self.root.bind("Q", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())


# ─────────────────────────────── запуск ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digit crop annotator")
    parser.add_argument("--crops_dir", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    root = tk.Tk()
    root.withdraw()

    # стартовый диалог — если аргументы не переданы
    crops_dir = args.crops_dir
    model_path = args.model

    if not crops_dir or not Path(crops_dir).is_dir():
        dlg = SetupDialog(root)
        if dlg._cancelled:
            print("Отмена.")
            sys.exit(0)
        crops_dir = dlg.crops_dir
        model_path = dlg.model_path

    model, transform = load_model(model_path)
    if model is None and model_path:
        print("[info] Модель не загружена — работаем без предсказаний.")

    root.deiconify()
    AnnotatorApp(root, crops_dir, model, transform)
    root.mainloop()


if __name__ == "__main__":
    main()