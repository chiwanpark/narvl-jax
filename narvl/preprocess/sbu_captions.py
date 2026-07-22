import json
from dataclasses import dataclass
from pathlib import Path
from tarfile import open as open_tar
from typing import Annotated, BinaryIO, cast
from urllib.request import Request, urlopen

import msgpack
import typer
from array_record.python import array_record_module  # ty: ignore[unresolved-import]

from narvl.preprocess.common import MAX_RECORDS_PER_FILE, ParallelPipeline, Row, make_caption_row, output_paths

DEFAULT_WORKERS = 32
DOWNLOAD_TIMEOUT = 120
USER_AGENT = "narvl-jax"
CAPTION_FILE = "sbu-captions-all.json"
app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class CaptionExample:
    url: str
    caption: str


def load_captions(input_path: Path | str) -> list[CaptionExample]:
    with open_tar(input_path, "r:gz") as archive:
        source = cast(BinaryIO, archive.extractfile(CAPTION_FILE))
        with source:
            data = json.load(source)
    return [CaptionExample(url, caption) for url, caption in zip(data["image_urls"], data["captions"], strict=True)]


def _process_example(example: CaptionExample) -> Row | None:
    try:
        request = Request(example.url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            return make_caption_row(response.read(), [example.caption])
    except Exception:
        return None


def _finalize_train(output_dir: Path, temporary_paths: list[Path], record_count: int) -> None:
    if not temporary_paths:
        writer = array_record_module.ArrayRecordWriter(str(output_dir / "train.array_record"), "group_size:1")
        writer.close()
        return

    for temporary_path, output_path in zip(
        temporary_paths,
        output_paths(output_dir, "train", record_count, MAX_RECORDS_PER_FILE),
        strict=True,
    ):
        temporary_path.replace(output_path)


def _write_train(
    output_dir: Path,
    examples: list[CaptionExample],
    workers: int = DEFAULT_WORKERS,
) -> int:
    temporary_paths: list[Path] = []
    writer = None
    record_count = 0
    skipped_count = 0
    try:
        with ParallelPipeline(workers) as pipeline:
            with typer.progressbar(length=len(examples), label="Converting train") as progress:
                for row in pipeline.imap_unordered(_process_example, examples):
                    progress.update(1)
                    if row is None:
                        skipped_count += 1
                        continue

                    if record_count % MAX_RECORDS_PER_FILE == 0:
                        if writer is not None:
                            writer.close()
                            writer = None
                        temporary_path = output_dir / f".train-{len(temporary_paths):05d}.array_record"
                        temporary_paths.append(temporary_path)
                        writer = array_record_module.ArrayRecordWriter(str(temporary_path), "group_size:1")
                    if writer is None:
                        raise RuntimeError("ArrayRecord writer was not initialized")
                    writer.write(msgpack.packb(row, use_bin_type=True))
                    record_count += 1
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    _finalize_train(output_dir, temporary_paths, record_count)
    return skipped_count


def preprocess(
    input_path: Path | str,
    output_dir: Path | str,
    workers: int = DEFAULT_WORKERS,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    skipped_count = _write_train(output_path, load_captions(input_path), workers)
    if skipped_count:
        typer.echo(f"Skipped {skipped_count} images that could not be downloaded or decoded.", err=True)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(help="Path to the SBU captions tar.gz archive")],
    output_dir: Annotated[Path, typer.Argument(help="Directory for ArrayRecord files")],
    workers: Annotated[int, typer.Option(min=1, help="Pipeline workers")] = DEFAULT_WORKERS,
) -> None:
    preprocess(input_path, output_dir, workers)


if __name__ == "__main__":
    app()
