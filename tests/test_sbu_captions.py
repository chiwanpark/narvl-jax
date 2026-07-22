import json
import msgpack
from collections.abc import Callable, Iterable, Iterator
from io import BytesIO
from pathlib import Path
from tarfile import TarInfo, open as open_tar
from typing import Self, cast

from array_record.python import array_record_module  # ty: ignore[unresolved-import]
from PIL import Image
from pytest import MonkeyPatch

from narvl.preprocess import sbu_captions
from narvl.preprocess.common import Row, make_caption_row


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (400, 200), (255, 0, 0)).save(output, format="JPEG")
    return output.getvalue()


def _write_archive(path: Path, data: object) -> None:
    content = json.dumps(data).encode()
    info = TarInfo("sbu-captions-all.json")
    info.size = len(content)
    with open_tar(path, "w:gz") as archive:
        archive.addfile(info, BytesIO(content))


def _rows(path: Path) -> list[Row]:
    reader = array_record_module.ArrayRecordReader(str(path))
    try:
        return [cast(Row, msgpack.unpackb(record, raw=False)) for record in reader.read_all()]
    finally:
        reader.close()


class FakeParallelPipeline:
    def __init__(self, _workers: int) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def imap_unordered(
        self,
        function: Callable[[sbu_captions.CaptionExample], Row | None],
        examples: Iterable[sbu_captions.CaptionExample],
    ) -> Iterator[Row | None]:
        return map(function, examples)


def test_preprocess_downloads_sbu_images(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    archive_path = tmp_path / "sbu-captions.tar.gz"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {
            "image_urls": ["https://example.com/image.jpg", "https://example.com/missing.jpg"],
            "user_ids": ["user", "user"],
            "captions": ["A red image", "Missing image"],
        },
    )

    def process_example(example: sbu_captions.CaptionExample) -> Row | None:
        if example.url.endswith("missing.jpg"):
            return None
        return make_caption_row(_jpeg(), [example.caption])

    monkeypatch.setattr(sbu_captions, "ParallelPipeline", FakeParallelPipeline)
    monkeypatch.setattr(sbu_captions, "_process_example", process_example)

    sbu_captions.preprocess(archive_path, output_dir, workers=2)

    rows = _rows(output_dir / "train.array_record")
    assert len(rows) == 1
    assert rows[0]["type"] == "caption"
    assert rows[0]["input_text"] is None
    assert rows[0]["outputs"] == ["A red image"]
    assert rows[0]["best_output_idx"] == 0
    with Image.open(BytesIO(rows[0]["input_img"])) as image:
        assert image.size == (256, 256)


def test_write_train_shards_processed_records(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    row: Row = {
        "type": "caption",
        "input_img": b"image",
        "input_text": None,
        "outputs": ["caption"],
        "best_output_idx": 0,
    }
    examples = [
        sbu_captions.CaptionExample("https://example.com/one.jpg", "one"),
        sbu_captions.CaptionExample("https://example.com/two.jpg", "two"),
        sbu_captions.CaptionExample("https://example.com/three.jpg", "three"),
    ]

    monkeypatch.setattr(sbu_captions, "MAX_RECORDS_PER_FILE", 2)
    monkeypatch.setattr(sbu_captions, "ParallelPipeline", FakeParallelPipeline)
    monkeypatch.setattr(sbu_captions, "_process_example", lambda _: row)

    assert sbu_captions._write_train(tmp_path, examples, workers=2) == 0

    paths = sorted(tmp_path.glob("*.array_record"))
    assert [path.name for path in paths] == [
        "train-00000-of-00002.array_record",
        "train-00001-of-00002.array_record",
    ]
    assert len(_rows(paths[0])) == 2
    assert len(_rows(paths[1])) == 1
