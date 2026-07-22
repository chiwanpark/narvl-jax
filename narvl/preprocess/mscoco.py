import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import msgpack
import typer
from array_record.python import array_record_module  # ty: ignore[unresolved-import]

from narvl.preprocess.common import (
    DEFAULT_WORKERS,
    MAX_RECORDS_PER_FILE,
    ParallelPipeline,
    output_paths,
    process_caption_row,
)

SPLITS = {"train": "train", "restval": "train", "val": "valid", "test": "test"}
OUTPUT_SPLITS = ("train", "valid", "test")
app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class CaptionExample:
    image_dir: str
    image_name: str
    captions: list[str]

    @property
    def image_path(self) -> str:
        return f"{self.image_dir}/{self.image_name}"


def load_captions(input_dir: Path | str) -> dict[str, list[CaptionExample]]:
    caption_path = Path(input_dir) / "caption_datasets.zip"
    with ZipFile(caption_path) as archive:
        with archive.open("dataset_coco.json") as source:
            data = json.load(source)

    examples = {split: [] for split in OUTPUT_SPLITS}
    for image in data["images"]:
        if (split := SPLITS.get(image["split"])) is not None:
            examples[split].append(
                CaptionExample(
                    image["filepath"], image["filename"], [sentence["raw"] for sentence in image["sentences"]]
                )
            )

    for split_examples in examples.values():
        split_examples.sort(key=lambda example: example.image_path)
    return examples


def _load_image(archives: dict[str, ZipFile], input_dir: Path, example: CaptionExample) -> bytes:
    archive = archives.get(example.image_dir)
    if archive is None:
        archive = ZipFile(input_dir / f"{example.image_dir}.zip")
        archives[example.image_dir] = archive
    return archive.read(example.image_path)


def _output_paths(output_dir: Path, split: str, record_count: int) -> list[Path]:
    return output_paths(output_dir, split, record_count, MAX_RECORDS_PER_FILE)


def _write_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    examples: list[CaptionExample],
    workers: int = DEFAULT_WORKERS,
) -> None:
    archives: dict[str, ZipFile] = {}
    output_paths = _output_paths(output_dir, split, len(examples))
    writer = None
    try:
        if not examples:
            writer = array_record_module.ArrayRecordWriter(str(output_paths[0]), "group_size:1")
            return

        with ParallelPipeline(workers) as pipeline:
            with typer.progressbar(length=len(examples), label=f"Converting {split}") as progress:
                input_data = ((_load_image(archives, input_dir, example), example.captions) for example in examples)
                for index, row in enumerate(pipeline.imap_unordered(process_caption_row, input_data)):
                    progress.update(1)
                    shard_index = index // MAX_RECORDS_PER_FILE
                    if index % MAX_RECORDS_PER_FILE == 0:
                        if writer is not None:
                            writer.close()
                        writer = array_record_module.ArrayRecordWriter(str(output_paths[shard_index]), "group_size:1")
                    if writer is None:
                        raise RuntimeError("ArrayRecord writer was not initialized")
                    writer.write(msgpack.packb(row, use_bin_type=True))
    finally:
        if writer is not None:
            writer.close()
        for archive in archives.values():
            archive.close()


def preprocess(
    input_dir: Path | str,
    output_dir: Path | str,
    workers: int = DEFAULT_WORKERS,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, examples in load_captions(input_path).items():
        _write_split(input_path, output_path, split, examples, workers)


@app.command()
def main(
    input_dir: Annotated[Path, typer.Argument(help="Directory containing the MSCOCO ZIP archives")],
    output_dir: Annotated[Path, typer.Argument(help="Directory for ArrayRecord files")],
    workers: Annotated[int, typer.Option(min=1, help="Pipeline workers")] = DEFAULT_WORKERS,
) -> None:
    preprocess(input_dir, output_dir, workers)


if __name__ == "__main__":
    app()
