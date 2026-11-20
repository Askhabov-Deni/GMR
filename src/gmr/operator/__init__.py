"""
src/gmr/operator — окно оператора (запуск: `python program2.py`).

Правила и файлы — без Tk:
  photos.py       — очереди: фото на разбор (question/) и на проверку (plus/, minus/)
  decisions.py    — исправление на «Проверке»: что поменяется в базе
  hints.py        — похожие номера в базе, выбор абонента
  files.py        — «Серийник в базе с ошибкой», раскладка и подпись фото
  markup.py       — тихая разметка исправлений для дообучения моделей
  settings.py     — settings.json: оператор и папка месяца
  model_bundle.py — модели в окне
Экраны (Tk):
  window.py        — главное окно: вход, месяц, меню, вкладки
  edit_screen.py   — разбор одного фото
  verify_screen.py — проверка одного фото
  dialogs.py, widgets.py, common.py — диалоги, поля показаний, цвета и журнал

Запись в базу месяца — src/gmr/application/operator.py (OperatorSession),
стык с автоматической обработкой — docs/contract_reader_program2.md.
"""
