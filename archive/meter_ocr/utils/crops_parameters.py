from pathlib import Path
from collections import Counter
from statistics import mean, median, variance, stdev, mode, StatisticsError

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

CROPS_DIR = Path(r"C:\AD\GSM\database\meter_ocr_data\cnn_dataset_gold")

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p: float):
    """
    Линейная интерполяция процентиля.
    p: 0..100
    """
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p / 100
    f = int(k)
    c = f + 1

    if c >= len(values):
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def calc_stats(values):
    if not values:
        return {}

    counter = Counter(values)

    try:
        mode_value = mode(values)
    except StatisticsError:
        mode_value = None

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "mode": mode_value,
        "variance": variance(values) if len(values) > 1 else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
    }


def print_stats(name, stats, integer=False):
    print(f"\n{name}")
    print("-" * 50)

    for key, value in stats.items():
        if integer and isinstance(value, (int, float)):
            print(f"{key:>10}: {value:.2f}")
        else:
            print(f"{key:>10}: {value}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not CROPS_DIR.exists():
        print(f"[ERROR] Папка не найдена: {CROPS_DIR}")
        return

    files = [
        p
        for p in CROPS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print(f"Папка: {CROPS_DIR.resolve()}")
    print(f"Найдено файлов: {len(files)}")

    if not files:
        return

    widths = []
    heights = []
    aspect_ratios = []
    areas = []

    bad_files = []

    for path in files:
        try:
            with Image.open(path) as img:
                width, height = img.size

            widths.append(width)
            heights.append(height)

            aspect_ratios.append(width / height)
            areas.append(width * height)

        except Exception as e:
            bad_files.append((path, str(e)))

    # --------------------------------------------------------
    # BASIC STATS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("РАЗМЕРЫ КРОПОВ")
    print("=" * 70)

    print_stats("WIDTH", calc_stats(widths), integer=True)
    print_stats("HEIGHT", calc_stats(heights), integer=True)

    print_stats("ASPECT RATIO (W/H)", calc_stats(aspect_ratios))

    print_stats("AREA (pixels)", calc_stats(areas), integer=True)

    # --------------------------------------------------------
    # UNIQUE SIZES
    # --------------------------------------------------------

    sizes = list(zip(widths, heights))
    size_counter = Counter(sizes)

    print("\n" + "=" * 70)
    print("УНИКАЛЬНЫЕ РАЗМЕРЫ")
    print("=" * 70)

    print(f"Уникальных размеров: {len(size_counter)}")

    print("\nСамые частые размеры:")

    for (w, h), count in size_counter.most_common(20):
        percentage = count / len(sizes) * 100

        print(
            f"  {w:4d} x {h:<4d} "
            f"{count:7d} шт. "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # BAD FILES
    # --------------------------------------------------------

    if bad_files:
        print("\n" + "=" * 70)
        print("ОШИБКИ")
        print("=" * 70)

        print(f"Не удалось прочитать: {len(bad_files)}")

        for path, error in bad_files[:20]:
            print(f"\n{path}")
            print(f"  {error}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("КРАТКИЙ ИТОГ")
    print("=" * 70)

    print(f"Успешно обработано: {len(widths)}")

    if widths:
        print(
            f"Средний размер: "
            f"{mean(widths):.2f} x {mean(heights):.2f}"
        )

        print(
            f"Медианный размер: "
            f"{median(widths):.2f} x {median(heights):.2f}"
        )

        print(
            f"Диапазон ширины: "
            f"{min(widths)} - {max(widths)}"
        )

        print(
            f"Диапазон высоты: "
            f"{min(heights)} - {max(heights)}"
        )

        print(
            f"Средний aspect ratio: "
            f"{mean(aspect_ratios):.3f}"
        )


if __name__ == "__main__":
    main()