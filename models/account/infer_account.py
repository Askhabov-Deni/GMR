"""
Инференс модели лицевого счёта по надписи маркером.

  python models/account/infer_account.py --checkpoint <best.pt> --image <кроп>
  python models/account/infer_account.py --checkpoint <best.pt> --image_dir <папка кропов>

Без словаря — только «жадное» чтение (что написано). Со словарём
(predict(crop, lexicon)) — какой лицевой счёт из таблицы это скорее всего и
насколько уверенно (models/ctc_lexicon.py).
"""
import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from .config_account import LEN_WINDOW, MAX_EDITS, MIN_CONFIDENCE
    from .dataset_account import prepare_input, read_gray
    from .model_account import load_checkpoint
    from ..ctc_lexicon import CompiledLexicon, greedy_decode, match
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.account.config_account import LEN_WINDOW, MAX_EDITS, MIN_CONFIDENCE
    from models.account.dataset_account import prepare_input, read_gray
    from models.account.model_account import load_checkpoint
    from models.ctc_lexicon import CompiledLexicon, greedy_decode, match


_EXTRA_MAX = 8        # «дополнить словарь» — если новых групп не больше стольких


def _extra_over(base: Mapping, groups: Mapping) -> Optional[dict]:
    """groups = base и в конце ещё несколько групп (тот же порядок, те же
    варианты) → эти группы; иначе None."""
    if not 0 < len(groups) - len(base) <= _EXTRA_MAX:
        return None
    rest = iter(groups.items())
    for (k, v), (k2, v2) in zip(base.items(), rest):     # zip берёт из base первым — лишнего из rest не съест
        if k != k2 or (v is not v2 and list(v) != list(v2)):
            return None
    return dict(rest)


class AccountInferer:
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model, self.meta = load_checkpoint(checkpoint_path, self.device)
        self.alphabet = self.meta["alphabet"]
        self.blank = self.model.blank
        self.img_h = self.meta["preprocess"]["img_h"]
        self.img_w = self.meta["preprocess"]["img_w"]
        self._compiled: list[tuple[object, CompiledLexicon]] = []     # последние словари, свежий — первый

    @torch.no_grad()
    def log_probs(self, image) -> torch.Tensor:
        """(T, C) для одной картинки (путь или ndarray BGR/серый)."""
        img = read_gray(image) if isinstance(image, (str, Path)) else image
        x = torch.from_numpy(prepare_input(np.asarray(img), self.img_h, self.img_w))
        return self.model(x.unsqueeze(0).to(self.device))[:, 0, :].cpu()

    KEEP = 3          # сколько словарей помнить: словарь участка, всей таблицы и разовый

    def compile(self, groups: Mapping[str, Sequence[str]]) -> CompiledLexicon:
        """Словарь готовится один раз и запоминается (по объекту groups):
        таблица одна на весь прогон. Словарь «запомненный и ещё пара счетов в
        конце» ({**groups, счёт: …} — сверка со счётом серийника другого
        кода) не собирается заново, а дополняется (1.6.0: раньше он собирался
        заново и вытеснял основной — ещё полсекунды на следующем фото)."""
        for i, (g, lex) in enumerate(self._compiled):
            if g is groups:
                self._compiled.insert(0, self._compiled.pop(i))
                return lex
        lex = None
        for g, base in self._compiled:
            extra = _extra_over(g, groups)
            if extra is not None:
                lex = base.extended(extra)
                break
        if lex is None:
            lex = CompiledLexicon(groups, self.alphabet)
        self._compiled = [(groups, lex)] + self._compiled[:self.KEEP - 1]
        return lex

    def predict(self, image, groups: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
        """{"text", "text_conf", "group", "string", "confidence", "top"}.

        text / text_conf — жадное чтение и средняя уверенность его символов;
        group и дальше — только со словарём groups ({счёт: [варианты записи]})."""
        lp = self.log_probs(image)
        text = greedy_decode(lp, self.alphabet, self.blank)
        probs = lp.exp()
        best = probs.argmax(1)
        chars = [float(probs[t, i]) for t, i in enumerate(best.tolist())
                 if i != self.blank and (t == 0 or i != int(best[t - 1]))]
        res = {"text": text, "text_conf": sum(chars) / len(chars) if chars else 0.0,
               "group": None, "string": None, "confidence": 0.0, "top": []}
        if groups:
            m = match(lp, self.compile(groups), self.blank, len_window=LEN_WINDOW, max_edits=MAX_EDITS)
            res.update(group=m.group, string=m.string, confidence=m.confidence, top=m.top)
        return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Чтение надписи маркером (лицевой счёт)")
    p.add_argument("--checkpoint", required=True, help="best.pt из account_ocr/runs/crnn/<дата-время>")
    p.add_argument("--image", help="один кроп")
    p.add_argument("--image_dir", help="папка кропов")
    p.add_argument("--min_conf", type=float, default=MIN_CONFIDENCE)
    args = p.parse_args(argv)
    if not args.image and not args.image_dir:
        p.error("нужен --image или --image_dir")

    inferer = AccountInferer(args.checkpoint)
    paths = [Path(args.image)] if args.image else sorted(
        q for q in Path(args.image_dir).iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    good = 0
    for path in paths:
        r = inferer.predict(path)
        ok = r["text_conf"] >= args.min_conf
        good += ok
        print(f"{'✅' if ok else '❌'} {path.name:40s} → '{r['text']}' ({r['text_conf']:.3f})")
    print(f"\nУверенно: {good} из {len(paths)} (без словаря; со словарём — evaluate_account.py --table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
