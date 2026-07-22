import os
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.pool import Pool
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar, TypedDict

from PIL import Image, ImageOps

IMAGE_SIZE = 256
BLACK = (0, 0, 0)
MAX_RECORDS_PER_FILE = 10_000
DEFAULT_WORKERS = os.process_cpu_count() or 1


class Row(TypedDict):
    type: str
    input_img: bytes
    input_text: str | None
    outputs: list[str]
    best_output_idx: int


def resize_image(image: Image.Image, size: int = IMAGE_SIZE) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    letterboxed = Image.new("RGB", (size, size), color=BLACK)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    letterboxed.paste(resized, offset)

    output = BytesIO()
    letterboxed.save(output, format="JPEG")
    return output.getvalue()


def make_caption_row(image_data: bytes, captions: list[str]) -> Row:
    with Image.open(BytesIO(image_data)) as image:
        input_img = resize_image(image)
    return {
        "type": "caption",
        "input_img": input_img,
        "input_text": None,
        "outputs": captions,
        "best_output_idx": 0,
    }


def process_caption_row(input_data: tuple[bytes, list[str]]) -> Row:
    return make_caption_row(*input_data)


Input = TypeVar("Input")
Output = TypeVar("Output")


class ParallelPipeline:
    def __init__(self, workers: int = DEFAULT_WORKERS) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._pool = Pool(workers)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is None:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()

    def imap_unordered(self, function: Callable[[Input], Output], inputs: Iterable[Input]) -> Iterator[Output]:
        return self._pool.imap_unordered(function, inputs)


def output_paths(
    output_dir: Path,
    split: str,
    record_count: int,
    max_records_per_file: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    shard_count = max(1, (record_count + max_records_per_file - 1) // max_records_per_file)
    if shard_count == 1:
        return [output_dir / f"{split}.array_record"]
    return [
        output_dir / f"{split}-{shard_index:05d}-of-{shard_count:05d}.array_record"
        for shard_index in range(shard_count)
    ]
