import itertools
from pathlib import Path
from typing import Annotated, Sequence

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
    """Parse indices given as comma-separated integers and/or start:stop[:step] ranges.

    Examples:
        "0,10,20"
        "0:30:10"
        "0:20:10,30
    """
    indices: list[int] = []
    index_or_slice_iterator = itertools.chain.from_iterable(el.value.split(",") for el in tokens)
    for index_or_slice in index_or_slice_iterator:
        if index_or_slice.isnumeric():
            index = int(index_or_slice)
            indices.append(index)
            continue
        parts = list(map(int, index_or_slice.split(":")))
        indices.extend(range(*parts))
    return indices


Indices = Annotated[
    list[int],
    Parameter(
        converter=_parse_indices,
        n_tokens=1,
        help="Indices given as comma-separated integers and/or start:stop[:step] ranges.",
    ),
]
