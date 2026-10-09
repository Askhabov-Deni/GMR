"""
src/gmr/domain/preset.py — настройки распознавания месяца («пресет»;
решения владельца 2026-10-11: 2а — задаются при создании месяца, заполнены
как в прошлом месяце, меняются через «Месяц» → «Настройки распознавания…»,
действуют на следующие обработки).

Пресет хранится в базе месяца (meta "recognition", JSON) и накладывается на
PipelineConfig перед прогоном и в окне оператора. По умолчанию — всё как
было до пресетов. Неизвестные ключи и неверные значения в JSON не ломают
загрузку: берётся значение по умолчанию.
"""
import dataclasses
import json
from dataclasses import dataclass
from typing import Optional

from src.gmr.domain.config import PipelineConfig

# пропущенная цифра (детектор нашёл на одну меньше): что ставить на её место
MISSING_MODES = {
    "placeholder": "подставить «5»",
    "model": "прочитать моделью на этом месте",
    "operator": "отдать оператору",
}
# неуверенные последние цифры (прощённые позиции): чем заменять
FORGIVE_MODES = {
    "placeholder": "«0»",
    "model": "ответом модели",
}
FORGIVE_LAST = (0, 1, 2)
SERIAL_PADS = (0.0, 0.05, 0.1)


@dataclass(frozen=True)
class RecognitionPreset:
    missing_digit: str = "placeholder"
    forgive_last: int = 2
    forgive_with: str = "placeholder"
    serial_pad: float = 0.0
    account_marker: bool = False      # лицевой счёт по надписи (нужна модель — account_ocr_model)
    # этап 6b (PipelineConfig.drum_rule, first_digit_from_last, serial_by_table)
    drum_rule: bool = False
    first_from_last: bool = False
    serial_by_table: bool = False
    turn_photo: bool = False          # этап 7b (PipelineConfig.turn_if_nothing)

    @classmethod
    def from_json(cls, text: Optional[str]) -> "RecognitionPreset":
        """Из meta базы месяца; пусто или испорчено — по умолчанию."""
        default = cls()
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        def pick(key, allowed):
            value = data.get(key, getattr(default, key))
            return value if value in allowed else getattr(default, key)
        return cls(
            missing_digit=pick("missing_digit", MISSING_MODES),
            forgive_last=pick("forgive_last", FORGIVE_LAST),
            forgive_with=pick("forgive_with", FORGIVE_MODES),
            serial_pad=pick("serial_pad", SERIAL_PADS),
            account_marker=data.get("account_marker") is True,
            drum_rule=data.get("drum_rule") is True,
            first_from_last=data.get("first_from_last") is True,
            serial_by_table=data.get("serial_by_table") is True,
            turn_photo=data.get("turn_photo") is True,
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, sort_keys=True)

    def apply(self, config: PipelineConfig) -> PipelineConfig:
        """PipelineConfig с этими настройками. Лицевой счёт по надписи
        выключен — модель надписи не грузится (account_ocr_model = "")."""
        return dataclasses.replace(
            config,
            missing_digit_mode=self.missing_digit,
            ignore_last_digits=self.forgive_last,
            forgiven_digit_mode=self.forgive_with,
            serial_crop_pad=self.serial_pad,
            account_ocr_model=config.account_ocr_model if self.account_marker else "",
            drum_rule=self.drum_rule,
            first_digit_from_last=self.first_from_last,
            serial_by_table=self.serial_by_table,
            turn_if_nothing=self.turn_photo,
        )

    def describe(self) -> str:
        """Одной строкой — для итога месяца и журнала прогона."""
        tail = ("последние цифры не прощаются" if self.forgive_last == 0 else
                f"неуверенные последние {self.forgive_last} → {FORGIVE_MODES[self.forgive_with]}")
        return "; ".join([
            f"пропущенная цифра — {MISSING_MODES[self.missing_digit]}",
            tail,
            f"запас рамки серийника {int(self.serial_pad * 100)}%",
            "лицевой счёт по надписи — " + ("да" if self.account_marker else "нет"),
            "дополнительно — " + (", ".join(name for on, name in (
                (self.drum_rule, "правило барабана"),
                (self.first_from_last, "старшая цифра по прошлому показанию"),
                (self.serial_by_table, "серийник по таблице"),
                (self.turn_photo, "фото без находок — повернуть")) if on) or "нет"),
        ])
