import json
import msgpack
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Self, cast
from zipfile import ZIP_DEFLATED, ZipFile

from array_record.python import array_record_module  # ty: ignore[unresolved-import]
from PIL import Image
from pytest import MonkeyPatch

from narvl.preprocess import mscoco
from narvl.preprocess.common import Row
from narvl.preprocess.mscoco import MAX_RECORDS_PER_FILE, _output_paths, preprocess


def _jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG")
    return output.getvalue()


def _write_images(path: Path, images: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, image in images.items():
            archive.writestr(name, image)


def _rows(path: Path) -> list[Row]:
    reader = array_record_module.ArrayRecordReader(str(path))
    try:
        return [cast(Row, msgpack.unpackb(record, raw=False)) for record in reader.read_all()]
    finally:
        reader.close()


def test_preprocess_writes_karpathy_splits(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    images = [
        {
            "split": "train",
            "filepath": "train2014",
            "filename": "z.jpg",
            "sentences": [{"raw": "train caption"}],
        },
        {
            "split": "restval",
            "filepath": "val2014",
            "filename": "restval.jpg",
            "sentences": [{"raw": "restval caption"}],
        },
        {
            "split": "val",
            "filepath": "val2014",
            "filename": "valid.jpg",
            "sentences": [{"raw": "valid caption 1"}, {"raw": "valid caption 2"}],
        },
        {
            "split": "test",
            "filepath": "val2014",
            "filename": "test.jpg",
            "sentences": [{"raw": "test caption"}],
        },
    ]
    with ZipFile(input_dir / "caption_datasets.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("dataset_coco.json", json.dumps({"images": images}))
    _write_images(input_dir / "train2014.zip", {"train2014/z.jpg": _jpeg((400, 200), (255, 0, 0))})
    _write_images(
        input_dir / "val2014.zip",
        {
            "val2014/restval.jpg": _jpeg((200, 400), (0, 255, 0)),
            "val2014/valid.jpg": _jpeg((200, 400), (0, 0, 255)),
            "val2014/test.jpg": _jpeg((200, 400), (255, 255, 0)),
        },
    )
    _write_images(input_dir / "test2014.zip", {})

    preprocess(input_dir, output_dir, workers=2)

    train_rows = _rows(output_dir / "train.array_record")
    valid_rows = _rows(output_dir / "valid.array_record")
    test_rows = _rows(output_dir / "test.array_record")
    assert sorted(row["outputs"] for row in train_rows) == [["restval caption"], ["train caption"]]
    assert valid_rows[0]["outputs"] == ["valid caption 1", "valid caption 2"]
    assert test_rows[0]["outputs"] == ["test caption"]

    for row in [*train_rows, *valid_rows, *test_rows]:
        assert row["type"] == "caption"
        assert row["input_text"] is None
        assert row["best_output_idx"] == 0
        with Image.open(BytesIO(row["input_img"])) as image:
            assert image.size == (256, 256)

    train_row = next(row for row in train_rows if row["outputs"] == ["train caption"])
    with Image.open(BytesIO(train_row["input_img"])) as image:
        assert image.getpixel((128, 0)) == (0, 0, 0)


def test_write_split_shards_large_datasets(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    paths = _output_paths(tmp_path, "train", MAX_RECORDS_PER_FILE + 1)
    examples = [mscoco.CaptionExample("train2014", "image.jpg", ["caption"])] * (MAX_RECORDS_PER_FILE + 1)
    row: Row = {
        "type": "caption",
        "input_img": b"image",
        "input_text": None,
        "outputs": ["caption"],
        "best_output_idx": 0,
    }

    class FakeParallelPipeline:
        def __init__(self, _workers: int) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def imap_unordered(
            self,
            _function: object,
            input_data: Iterable[tuple[bytes, list[str]]],
        ) -> Iterator[Row]:
            for _ in input_data:
                yield row

    monkeypatch.setattr(mscoco, "_load_image", lambda *_: b"")
    monkeypatch.setattr(mscoco, "ParallelPipeline", FakeParallelPipeline)

    mscoco._write_split(tmp_path, tmp_path, "train", examples)

    assert [path.name for path in paths] == [
        "train-00000-of-00002.array_record",
        "train-00001-of-00002.array_record",
    ]
    assert len(_rows(paths[0])) == MAX_RECORDS_PER_FILE
    assert len(_rows(paths[1])) == 1
