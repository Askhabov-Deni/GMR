"""
CRNN + CTC для надписи маркером (лицевой счёт на корпусе счётчика).

Вход:  (B, 1, H, W) — серое изображение после prepare_input (dataset_account.py)
Выход: (T, B, len(alphabet) + 1) — log-softmax, blank — последний индекс,
       T = W / 4.

Отличия от models/crnn/model_crnn.py (серийник):
  - один канал (цвет надписи не важен, вход в 3 раза меньше);
  - высота не усредняется в конце, а складывается в признаки шага — у
    рукописных цифр важна форма по вертикали (1/7, 3/8, 5/6);
  - LSTM вместо GRU (обычный выбор для рукописного текста);
  - чекпоинт описывает сам себя: архитектура, размер входа и алфавит
    хранятся в файле. Изменение этого файла или config_account.py не ломает
    старые веса, а инференс не может взять «не тот» размер входа
    (ловушка из docs/GUIDE.md, раздел 6, здесь закрыта).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "gmr-account-crnn/1"


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class AccountCRNN(nn.Module):
    def __init__(self, img_h: int = 48, n_symbols: int = 10, rnn_hidden: int = 128,
                 rnn_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        if img_h % 16:
            raise ValueError(f"img_h должна делиться на 16, а не {img_h}")
        self.arch = dict(img_h=img_h, n_symbols=n_symbols, rnn_hidden=rnn_hidden,
                         rnn_layers=rnn_layers, dropout=dropout)
        self.blank = n_symbols
        self.cnn = nn.Sequential(
            _block(1, 32), nn.MaxPool2d(2, 2),                      # H/2,  W/2
            _block(32, 64), nn.MaxPool2d(2, 2),                     # H/4,  W/4
            _block(64, 128), _block(128, 128), nn.MaxPool2d((2, 1)),  # H/8,  W/4
            _block(128, 256), _block(256, 256), nn.MaxPool2d((2, 1)),  # H/16, W/4
            nn.Dropout2d(dropout / 2),
        )
        feat = 256 * (img_h // 16)
        self.proj = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.rnn = nn.LSTM(256, rnn_hidden, num_layers=rnn_layers, bidirectional=True,
                           dropout=dropout if rnn_layers > 1 else 0.0)
        self.fc = nn.Linear(2 * rnn_hidden, n_symbols + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                              # (B, C, h, T)
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)  # (T, B, C*h)
        f = self.proj(f)
        f, _ = self.rnn(f)
        return F.log_softmax(self.fc(f), dim=2)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_IMPOSSIBLE = 1e4       # «−log P» варианта, который не влезает в выход модели


def multi_ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor,
                   owners: torch.Tensor, blank: int) -> torch.Tensor:
    """CTC, когда у картинки несколько допустимых меток (надпись — любое из
    написаний счёта, dataset_account): −log Σ P(вариант | картинка) по её
    вариантам. Модель, которая уже читает цифры, сама «выбирает» написание,
    которое видит на картинке; с нуля так не учится — нужна разминка
    (synthetic.py, train_account). Один вариант — обычный CTC. Масштаб — как
    у обычного CTC (на символ), среднее по пачке. owners[k] — номер картинки
    варианта k."""
    t, b = log_probs.shape[:2]
    lp = log_probs[:, owners, :]
    input_lengths = torch.full((lp.shape[1],), t, dtype=torch.long, device=log_probs.device)
    nll = F.ctc_loss(lp, targets, input_lengths, target_lengths, blank=blank,
                     reduction="none", zero_infinity=False)
    nll = torch.nan_to_num(nll, nan=_IMPOSSIBLE, posinf=_IMPOSSIBLE)      # вариант не влезает в T шагов
    neg = -nll
    peak = torch.full((b,), float("-inf"), device=neg.device).scatter_reduce(
        0, owners, neg, reduce="amax", include_self=True)
    total = torch.zeros(b, device=neg.device).index_add(0, owners, torch.exp(neg - peak[owners]))
    marginal = -(torch.log(total) + peak)                                  # −log Σ P(вариант)
    return marginal.mean() / target_lengths.float().mean()


def save_checkpoint(path, model: AccountCRNN, preprocess: dict, alphabet: str, **extra) -> None:
    """Веса + всё, что нужно, чтобы прочитать картинку так же, как при обучении."""
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "arch": dict(model.arch),
        "preprocess": dict(preprocess),
        "alphabet": alphabet,
        "model_state": model.state_dict(),
        **extra,
    }, str(path))


def load_checkpoint(path, device="cpu") -> tuple[AccountCRNN, dict]:
    """(модель в режиме eval, метаданные чекпоинта без весов)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {path}")
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: это не чекпоинт модели лицевого счёта ({CHECKPOINT_FORMAT})")
    model = AccountCRNN(**ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "optimizer_state")}
    return model, meta
