# Copyright 2023, European Centre for Medium Range Weather Forecasts.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from typing import Sequence

from plotly import colors as _pc


def recursive_dict_update(original_dict, update_dict):
    """
    Recursively update a dictionary with keys and values from another dictionary.

    Parameters
    ----------
    original_dict : dict
        The original dictionary to be updated (in place).
    update_dict : dict
        The dictionary containing keys to be updated in the original dictionary.
    """
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            original_dict[k] = recursive_dict_update(original_dict.get(k, {}), v)
        else:
            original_dict[k] = v
    return original_dict


def list_to_human(iterable, conjunction="and", oxford_comma=False):
    """
    Convert an iterable to a human-readable string.

    Parameters
    ----------
    iterable : list or tuple
        The list of strings to convert to a single natural language string.
    conjunction : str, optional
        The conjunction with which to join the last two elements of the list,
        for example "and" (default).
    oxford_comma : bool, optional
        If `True`, an "Oxford comma" will be added before the conjunction when
        there are three or more elements in the list. Default is `False`.
    """
    list_of_strs = [str(item) for item in iterable]

    if len(list_of_strs) > 2:
        list_of_strs = [", ".join(list_of_strs[:-1]), list_of_strs[-1]]
        if oxford_comma:
            list_of_strs[0] += ","

    return f" {conjunction} ".join(list_of_strs)


def discrete_scale(
    colors: str | Sequence[str],
    bounds: Sequence[float] | None = None,
    n_bins: int = 12,
) -> list[tuple[float, str]]:
    """
    Return a *step* colourscale with flat bands, either:

    1. By explicit `bounds=[b0,b1,…,bN]` (creates N intervals), or
    2. By specifying `n_bins=k` (divides the range [0,1] into k equal bands).

    `colors` can be:
      - a named Plotly colourscale (str), or
      - a list of k hex/RGB strings (len must match #intervals).
    """
    # mode checks
    if bounds is None and n_bins is None:
        raise ValueError("Must supply either bounds or n_bins")

    # determine bins & normalization domain
    if bounds is not None:
        if len(bounds) < 2 or any(
            bounds[i] >= bounds[i + 1] for i in range(len(bounds) - 1)
        ):
            raise ValueError(
                "bounds must be a strictly increasing sequence of ≥2 values"
            )
        lo0, hiN = bounds[0], bounds[-1]
        intervals = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        k = len(intervals)
    else:
        # equal-spaced bins on [0,1]
        if n_bins < 1:
            raise ValueError("n_bins must be ≥1")
        lo0, hiN = 0.0, 1.0
        k = n_bins
        intervals = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]

    # get colours
    if isinstance(colors, str):
        # sample from named scale
        rgb = _pc.sample_colorscale(colors, samplepoints=k)
    else:
        rgb = colors  # type: ignore

    # build stepped colourscale
    out: list[tuple[float, str]] = []
    for (lo, hi), col in zip(intervals, rgb):
        # if using bounds mode, normalize lo/hi into [0,1]
        if bounds is not None:
            lo, hi = (lo - lo0) / (hiN - lo0), (hi - lo0) / (hiN - lo0)
        out.extend([(lo, col), (hi, col)])

    return out
