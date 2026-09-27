"""
Архитектура CNN-классификатора цифр (0-9).

Лёгкая сеть с нуля: три свёрточных блока + полносвязный классификатор.
После обучения экспортируется в TorchScript — класс DigitCNN нужен
только здесь и в train_cnn.py; для инференса достаточно .pt-файла.
"""

import torch
import torch.nn as nn

from config_cnn import IMG_SIZE, IMG_WIDTH, NUM_CLASSES, DROPOUT, DEVICE


class DigitCNN(nn.Module):
    """
    Input:  (B, 3, IMG_SIZE, IMG_WIDTH)  →  по умолчанию (B, 3, 96, 48)
    Output: (B, NUM_CLASSES)
    """

    # Параметры пулинга, от которых зависит размер входа классификатора.
    # Цифра на кропе выше, чем шире (typical AR W/H ~ 0.41-0.46), поэтому
    # выходная форма пулинга тоже выше, чем шире (4 x 2), а не наоборот —
    # это сохраняет пропорцию входа (IMG_SIZE:IMG_WIDTH = 96:48 = 2:1)
    # вплоть до самого классификатора. Число фич то же самое (4*2=8),
    # так что классификатор и риск переобучения не меняются.
    _POOL_H = 4
    _POOL_W = 2

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self._POOL_H, self._POOL_W)),
        )

        in_features = 64 * self._POOL_H * self._POOL_W  # 64 * 4 * 2 = 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _format_val_acc(ckpt: dict) -> str:
        """val_acc может отсутствовать в чекпоинте — тогда get() вернёт None,
        и ':.4f' на None/строке упадёт с ValueError. Форматируем безопасно."""
        val_acc = ckpt.get("val_acc")
        return f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "неизвестно"

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: torch.device = DEVICE) -> "DigitCNN":
        """
        Загружает модель из чекпоинта.
        """
        model = cls().to(device)

        # weights_only=True рекомендуется в PyTorch >= 2.0 для безопасности
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Проверяем ключи, под которыми могли быть сохранены веса
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[CNN] Загружена из {checkpoint_path} (val_acc={cls._format_val_acc(ckpt)})")
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"[CNN] Загружена из {checkpoint_path} (val_acc={cls._format_val_acc(ckpt)})")
        else:
            # Если чекпоинт — это просто state_dict без обёртки
            model.load_state_dict(ckpt)
            print(f"[CNN] Загружена из {checkpoint_path}")

        model.eval()
        return model


if __name__ == "__main__":
    model = DigitCNN()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_WIDTH)
    
    # ВАЖНО: Переключаем в eval перед прогонкой батча размером 1
    model.eval()
    
    out = model(dummy)
    print(f"Output shape : {out.shape}")        # (1, 10)
    print(f"Parameters   : {model.num_parameters:,}")
    
    # Тест загрузки
    checkpoint_path = "meter_ocr/runs/cnn/runs/v2_gold/best.pth"
    try:
        model_loaded = DigitCNN.from_pretrained(checkpoint_path, device=DEVICE)
        out_loaded = model_loaded(dummy)
        print(f"Loaded output shape : {out_loaded.shape}")
    except FileNotFoundError:
        print(f"\n[WARNING] Файл {checkpoint_path} не найден. Пропускаем тест загрузки.")