"""
Инструмент быстрой разметки для CRNN.
Формат вывода: image.jpg → labels/image.txt (внутри: "12345")

Клавиши (как нумпад):
  q w e  →  7 8 9
  a s d  →  4 5 6
  z x c  →  1 2 3
  space  →  0
  Backspace → удалить последнюю
  Enter/→   → сохранить и далее
  ←         → назад
  Esc       → выход
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Маппинг клавиш → цифры
KEY_MAP = {
    'q': '7', 'w': '8', 'e': '9',
    'a': '4', 's': '5', 'd': '6',
    'z': '1', 'x': '2', 'c': '3',
    'space': '0'
}

class OCRLabeler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OCR Labeler | Gas Meter")
        self.root.configure(bg="#121218")
        self.root.geometry("850x620")
        self.root.resizable(False, False)

        self.img_dir = self._ask_dir("Выберите папку с ИЗОБРАЖЕНИЯМИ")
        if not self.img_dir: return

        self.lbl_dir = self.img_dir.parent / "labels"
        self.lbl_dir.mkdir(exist_ok=True)

        self.img_files = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        if not self.img_files:
            messagebox.showerror("Ошибка", "Изображения не найдены!"); return

        # 🔍 Ищем первое неразмеченное изображение
        self.idx = next(
            (i for i, f in enumerate(self.img_files) 
             if not (self.lbl_dir / f"{f.stem}.txt").exists()),
            0
        )

        self._setup_ui()
        self.root.bind("<Key>", self._on_key)
        self._show(self.idx)
        self.root.mainloop()

    def _ask_dir(self, title):
        d = filedialog.askdirectory(title=title)
        return Path(d) if d else None

    def _setup_ui(self):
        # Изображение
        self.canvas = tk.Label(self.root, bg="#121218")
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)

        # Поле ввода
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            self.root, textvariable=self.entry_var,
            font=("Consolas", 32, "bold"), justify="center",
            bg="#1a1a20", fg="#060706", insertbackground="#00ff88",
            state="readonly", bd=0, highlightthickness=0
        )
        self.entry.pack(pady=8)

        # Подсказка по клавишам
        hint_txt = (
            "  q w e → 7 8 9   |   a s d → 4 5 6   |   z x c → 1 2 3   |   space → 0\n"
            "  Backspace ← удалить  |  Enter/→ сохранить  |  ← назад  |  Esc выход"
        )
        self.hint = tk.Label(
            self.root, text=hint_txt, bg="#121218", fg="#8888aa",
            font=("Courier", 11), justify="left"
        )
        self.hint.pack(pady=2)

        # Прогресс
        self.info = tk.Label(
            self.root, text="", bg="#121218", fg="#8888aa", font=("Courier", 10)
        )
        self.info.pack(side="bottom", pady=6)

    def _show(self, idx):
        self.idx = max(0, min(idx, len(self.img_files) - 1))
        img_path = self.img_files[self.idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        # Масштабирование под окно
        img = Image.open(img_path)
        w, h = img.size
        scale = min(800 / w, 420 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)

        # Загрузка существующей метки
        current = lbl_path.read_text().strip() if lbl_path.exists() else ""
        self.entry_var.set(current)
        self.entry.config(state="normal")
        self.entry.focus_set()
        self.entry.config(state="readonly")

        # Статус
        labeled = sum(1 for f in self.img_files if (self.lbl_dir / f"{f.stem}.txt").exists())
        status = "✅ уже размечено" if lbl_path.exists() else "📝 ожидает"
        self.info.config(
            text=f"[{self.idx+1}/{len(self.img_files)}] {img_path.name} | "
                 f"Готово: {labeled}/{len(self.img_files)} | {status}"
        )

    def _save_and_next(self):
        txt = self.entry_var.get().strip()
        if txt and not txt.isdigit():
            self.root.bell()  # звуковой сигнал при ошибке
            return

        lbl_path = self.lbl_dir / f"{self.img_files[self.idx].stem}.txt"
        lbl_path.write_text(txt, encoding="utf-8")

        # Переход к следующему
        next_idx = self.idx + 1
        if next_idx < len(self.img_files):
            self._show(next_idx)
        else:
            messagebox.showinfo("Готово", "Все изображения обработаны!")
            self.root.destroy()

    def _on_key(self, e):
        k = e.keysym.lower()

        # Ввод цифр
        if k in KEY_MAP:
            current = self.entry_var.get()
            if len(current) < 10:  # защита от бесконечного ввода
                self.entry_var.set(current + KEY_MAP[k])
            return

        # Удаление
        if k == 'backspace':
            self.entry_var.set(self.entry_var.get()[:-1])
            return

        # Сохранить и далее
        if k in ('return', 'right'):
            self._save_and_next()
            return

        # Назад
        if k == 'left':
            self._show(self.idx - 1)
            return

        # Выход
        if k == 'escape':
            self.root.destroy()
            return

if __name__ == "__main__":
    OCRLabeler()