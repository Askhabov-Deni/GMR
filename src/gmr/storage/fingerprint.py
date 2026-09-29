"""
Отпечаток содержимого фото — по нему фото узнаётся в логе независимо от
имени файла и папки (решение владельца 2026-09-29, MIGRATION_STATUS.md).
"""
import hashlib

FINGERPRINT_LENGTH = 16  # первые 16 hex-символов SHA-256 = 64 бита


def photo_fingerprint(path: str) -> str:
    """SHA-256 по байтам файла, первые 16 hex-символов. Имя файла не участвует."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:FINGERPRINT_LENGTH]
