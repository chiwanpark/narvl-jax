from io import StringIO

from loguru import logger

from narvl import main


def test_main() -> None:
    output = StringIO()
    handler_id = logger.add(output, format="{message}")
    try:
        main()
    finally:
        logger.remove(handler_id)

    assert output.getvalue() == "Hello from narvl-jax!\n"
