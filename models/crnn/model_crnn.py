"""
CRNN + CTC для распознавания показаний счётчика.

Input:  (B, 3, H, W)  — размеры из config_crnn.py
Output: (T, B, 11)    — log-softmax, blank=10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK     = 10
N_CLASSES = 11  # цифры 0–9 + blank


class CRNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        cnn_out: int = 256,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            block(in_channels, 64),
            nn.MaxPool2d(2, 2),                   # → (64, H/2, W/2)

            block(64, 128),
            nn.MaxPool2d(2, 2),                   # → (128, H/4, W/4)

            block(128, 256),
            block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),         # → (256, H/8, W/4)  только высота

            block(256, cnn_out),
            nn.AdaptiveAvgPool2d((1, None)),      # → (cnn_out, 1, T)
        )

        self.rnn = nn.GRU(
            input_size=cnn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * 2, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).squeeze(2)   # (B, C, T)
        feat = feat.permute(2, 0, 1)    # (T, B, C)
        out, _ = self.rnn(feat)         # (T, B, 2*hidden)
        return F.log_softmax(self.fc(out), dim=2)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "CRNN":
        """
        Загружает модель из чекпоинта.

        Использование:
            model = CRNN.from_pretrained("best.pt", device="cuda")
        """
        model = cls().to(device)
        ckpt  = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            val_acc = ckpt.get("val_acc", "?")
            print(f"[CRNN] Загружен из {checkpoint_path} (val_acc={val_acc:.4f})")
        else:
            model.load_state_dict(ckpt)
            print(f"[CRNN] Загружен из {checkpoint_path}")
        model.eval()
        return model


# ── Loss ──────────────────────────────────────────────────────────────

_ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    T, B = log_probs.shape[:2]
    input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
    return _ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)


# ── Greedy decode ─────────────────────────────────────────────────────

def decode(log_probs: torch.Tensor) -> list[str]:
    """(T, B, C) → список строк."""
    indices = log_probs.argmax(2).permute(1, 0)  # (B, T)
    results = []
    for seq in indices:
        out, prev = [], -1
        for idx in seq.tolist():
            if idx != prev:
                out.append(idx)
            prev = idx
        results.append("".join(str(i) for i in out if i != BLANK))
    return results


@torch.no_grad()
def predict(model: CRNN, images: torch.Tensor) -> list[str]:
    model.eval()
    return decode(model(images))


# ── Greedy decode с уверенностью ─────────────────────────────────────

def decode_with_confidence(log_probs: torch.Tensor) -> list[dict]:
    """
    (T, B, C) → список словарей {"text": str, "details": [{"char": str, "conf": float}]}
    """
    probs   = torch.exp(log_probs)           # (T, B, C)
    indices = log_probs.argmax(2).permute(1, 0)  # (B, T)

    results = []
    for b in range(indices.shape[0]):
        seq       = indices[b]
        seq_probs = probs[:, b, :]  # (T, C)
        out_chars, out_details, prev = [], [], -1

        for t, idx in enumerate(seq.tolist()):
            if idx != prev:
                if idx != BLANK:
                    char = str(idx)
                    conf = seq_probs[t, idx].item()
                    out_chars.append(char)
                    out_details.append({"char": char, "conf": conf})
                prev = idx

        results.append({"text": "".join(out_chars), "details": out_details})

    return results


@torch.no_grad()
def predict_with_confidence(model: CRNN, images: torch.Tensor) -> list[dict]:
    """Аналог predict, но с детальной информацией об уверенности."""
    model.eval()
    return decode_with_confidence(model(images))
