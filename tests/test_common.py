from narvl.preprocess.common import ParallelPipeline


def _double(value: int) -> int:
    return value * 2


def test_parallel_pipeline_maps_unordered() -> None:
    with ParallelPipeline(workers=2) as pipeline:
        results = list(pipeline.imap_unordered(_double, range(4)))

    assert sorted(results) == [0, 2, 4, 6]
