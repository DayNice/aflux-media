from pathlib import Path
from typing import Annotated, Sequence, assert_never

import cyclopts
from cyclopts import Parameter, Token, validators

InputFile = Annotated[
    Path,
    Parameter(validator=validators.Path(exists=True, dir_okay=False)),
]

OutputFile = Annotated[
    Path,
    Parameter(validator=validators.Path(dir_okay=False)),
]

InputDir = Annotated[
    Path,
    Parameter(validator=validators.Path(exists=True, file_okay=False)),
]

OutputDir = Annotated[
    Path,
    Parameter(validator=validators.Path(file_okay=False)),
]


def _parse_indices[T](type_: type[T], tokens: Sequence[Token]) -> list[int]:
    """Parse indices given as space-separated integers and/or start:stop[:step] ranges.

    Examples:
        "0 10 20"
        "0:30:10"
        "0:20:10 30"
    """
    parsed_items: list[int | slice] = cyclopts.convert(list[int | slice], tokens)
    indices: list[int] = []
    for item in parsed_items:
        match item:
            case int():
                indices.append(item)
            case slice():
                if item.stop is None:
                    raise ValueError("Slice stop must be provided.")
                start = item.start if item.start is not None else 0
                step = item.step if item.step is not None else 1
                indices.extend(range(start, item.stop, step))
            case _:
                assert_never(item)
    return indices


Indices = Annotated[
    list[int],
    Parameter(
        converter=_parse_indices,
        help="Indices given as space-separated integers and/or start:stop[:step] ranges.",
    ),
]
