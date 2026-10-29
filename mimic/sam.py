"""Read SAM 3.1's answers: where is each object, in pixels.

A SAM reply (masks shortened) looks like this, for a photo and for a video alike:

    <0f>0<|box;x1=0;y1=809;x2=84;y2=909;w=1920;h=1080|><|mask;...|>,1<|box;...|><|mask;...|>
    <1f>0<|box;...|><|mask;...|>

- <Nf> starts frame N. A photo is frame 0.
- The number right before <|box is the match id ("0", then ",1", ",2" ...).
  In a video it is the track id: the same object keeps the same id in every frame.
- No match at all is an empty reply.
- We ignore the masks (outlines): the box centre is enough to locate a block.
"""
import re
from dataclasses import dataclass

from mimic import config, meta_api

# Mask text never contains < or > (checked on every saved reply, tests/unit/test_sam.py),
# so a mask runs from "<|mask;" to the next ">", or to the end if the reply was cut short.
MASK = re.compile(r"<\|mask;[^<>]*>?")
TOKEN = re.compile(r"<(\d+)f>|(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    w: int  # width of the whole image in pixels, not of the box
    h: int  # height of the whole image in pixels, not of the box
    id: int = 0  # match id in a photo, track id in a video

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def _tokens(text: str):
    """Yield ("frame", n) and ("box", Box) in the order they appear. Unfinished tokens are skipped."""
    for m in TOKEN.finditer(MASK.sub("", text)):
        if m.group(1) is not None:
            yield "frame", int(m.group(1))
        else:
            match_id, x1, y1, x2, y2, w, h = (int(m.group(i)) for i in range(2, 9))
            yield "box", Box(x1, y1, x2, y2, w, h, id=match_id)


def parse_boxes(text: str) -> list[Box]:
    """Every box in a photo reply, in the order SAM gave them."""
    return [item for kind, item in _tokens(text) if kind == "box"]


def parse_tracks(text: str) -> dict[int, list[tuple[int, Box]]]:
    """Video reply -> {track id: [(frame, box), ...]}.

    A frame marker only says which frame the next boxes belong to. A frame where SAM saw
    nothing is just a marker with no box after it, so it adds nothing.
    """
    tracks = {}
    frame = 0
    for kind, item in _tokens(text):
        if kind == "frame":
            frame = item
        else:
            tracks.setdefault(item.id, []).append((frame, item))
    return tracks


def find(image_path: str, name: str, use_cache: bool = True) -> list[Box]:
    """Ask SAM where `name` (e.g. "blue block") is in the photo. Cached and cost-counted by meta_api."""
    content = [{"type": "input_text", "text": name},
               {"type": "input_image", "image_url": meta_api.image_url(image_path)}]
    return parse_boxes(meta_api.call(config.SAM_MODEL, content, use_cache=use_cache))
