# Образцы символов надписей контролёров

`glyphs.npz` — образцы символов `0123456789покслПОКЛС` (src/gmr/overlay.py:
найти и прочитать надпись, набранную поверх фото в редакторе телефона).
Символ, которого нет в шрифте, пропущен (телефон рисует его системным).

Шрифты (свободные; в проект положены только образцы, не сами шрифты):

| шрифт | начертания | откуда похожая надпись | лицензия |
|---|---|---|---|
| Roboto | 400, 500 | Android | Apache 2.0 / OFL 1.1 |
| Noto Sans | 400, 500 | Android (Samsung, Xiaomi — близкие формы) | OFL 1.1 |
| Inter | 400, 500 | iPhone (близко к SF Pro) | OFL 1.1 |
| Arimo | 400, 500 | iPhone (метрики Helvetica) | Apache 2.0 / OFL 1.1 |
| Open Sans | 400 | редакторы фото | OFL 1.1 |
| Oswald, Calistoga, Courier Prime | 400 | стили текста WhatsApp | OFL 1.1 |

Источник — Google Fonts (пакеты `@expo-google-fonts/<шрифт>` в npm).
Пересобрать (нужны Pillow и fontTools):

```powershell
python -c "from src.gmr.overlay import make_bank; print(make_bank([r'<путь>\Roboto_400Regular.ttf', ...]))"
```
