"""Write a text note into an image file's own metadata, so a simulation's pictures say so even when renamed.

JPEG has a standard comment segment (marker FF FE) and PNG a standard text chunk (tEXt); both are ignored when the
picture is shown. OpenCV cannot write either, so the bytes are added here: right after the JPEG's first two bytes,
or right after the PNG's header chunk.
"""
import struct
import zlib
from pathlib import Path

JPEG_START = b"\xff\xd8"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def mark_jpeg(path, text: str) -> None:
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(JPEG_START):
        raise ValueError(f"{path} is not a JPEG.")
    note = text.encode("latin-1")
    segment = b"\xff\xfe" + struct.pack(">H", len(note) + 2) + note
    if data[2:2 + len(segment)] == segment:
        return  # already marked
    path.write_bytes(JPEG_START + segment + data[2:])


def jpeg_comment(path) -> str | None:
    data = Path(path).read_bytes()
    if data[2:4] != b"\xff\xfe":
        return None
    length = struct.unpack(">H", data[4:6])[0]
    return data[6:4 + length].decode("latin-1")


def mark_png(path, keyword: str, text: str) -> None:
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG.")
    body = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    chunk = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body))
    header_end = 8 + 4 + 4 + 13 + 4  # signature, then the IHDR chunk: length, type, 13 bytes, CRC
    path.write_bytes(data[:header_end] + chunk + data[header_end:])


def png_texts(path) -> dict:
    data, texts, at = Path(path).read_bytes(), {}, 8
    while at < len(data):
        length = struct.unpack(">I", data[at:at + 4])[0]
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        if kind == b"tEXt":
            keyword, _, text = body.partition(b"\x00")
            texts[keyword.decode("latin-1")] = text.decode("latin-1")
        at += 12 + length
    return texts
