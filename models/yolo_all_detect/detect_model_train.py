from ultralytics import YOLO

# Загрузка предобученной модели (или вашей, если продолжаете обучение)
model = YOLO('yolov8n.pt') 

model.train(
    data="database/meter_ocr_data/digits_detect/dataset/data.yaml",
    epochs=50,
    imgsz=480,
    batch=16,

    # === ГЕОМЕТРИЯ ===
    degrees=7.0,
    translate=0.1,
    scale=0.2,
    shear=1.0,
    perspective=0.0005,

    # === ОТРАЖЕНИЯ ===
    flipud=0.0,
    fliplr=0.0,

    # === ЦВЕТ И СВЕТ ===
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.4,

    mosaic=1.0,
    close_mosaic=20,

    # === ПУТИ ===
    project="meter_ocr/runs/yolo",
    name="digits_detect_v4",
    exist_ok=True,
    patience=20,
)