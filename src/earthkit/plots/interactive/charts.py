# Copyright 2024, European Centre for Medium Range Weather Forecasts.
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

from typing import Sequence

import numpy as np
from plotly.subplots import make_subplots

from earthkit.plots.interactive import bar, box, heatmap, inputs, line, times
from earthkit.plots.interactive.utils import discrete_scale

DEFAULT_LAYOUT = {
    "colorway": [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ],
    "hovermode": "x",
    "plot_bgcolor": "white",
    "xaxis": {
        "gridwidth": 1,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
    },
    "yaxis": {
        "linecolor": "black",
        "gridcolor": "#EEEEEE",
        "showgrid": True,
        "showline": True,
        "zeroline": False,
    },
    "height": 750,
    "showlegend": False,
}


class Chart:
    """
    A class for creating and managing multi-subplot interactive charts using Plotly.

    Parameters
    ----------
    rows : int, optional
        Number of rows in the chart grid. Default is 1.
    columns : int, optional
        Number of columns in the chart grid. Default is 1.
    **kwargs : dict
        Additional arguments passed to `plotly.subplots.make_subplots`.
    """

    def __init__(self, rows=None, columns=None, **kwargs):
        self._rows = rows
        self._columns = columns

        self._fig = None
        self._subplots = []
        self._subplots_kwargs = kwargs
        self._subplot_titles = None
        self._subplot_y_titles = None
        self._subplot_x_titles = None
        self._subplot_z_titles = None
        self._layout_override = dict()

    def set_subplot_titles(method):
        def wrapper(self, *args, **kwargs):
            # Always try to extract dataset and derive new titles
            new_titles = []
            try:
                if args:
                    ds = inputs.to_xarray(args[0])
                    new_titles = list(ds.data_vars)
                    units = [
                        ds[data_var].attrs.get("units", "") for data_var in ds.data_vars
                    ]
            except Exception:
                new_titles = []
                units = []

            # Initialize or append to existing subplot_titles
            if new_titles:
                if self._subplot_titles is None:
                    # First-time setup
                    self._subplot_titles = new_titles.copy()
                else:
                    # Append new ones
                    self._subplot_titles.extend(new_titles)

                # Determine where to place axis-specific titles
                if kwargs.get("y") is not None:
                    # x-axis titles provided explicitly
                    if self._subplot_x_titles is None:
                        self._subplot_x_titles = units.copy()
                    else:
                        self._subplot_x_titles.extend(units)
                else:
                    if method.__qualname__ == "Chart.heatmap":
                        y_dim = times.guess_non_time_dim(ds)
                        if self._subplot_y_titles is None:
                            self._subplot_y_titles = [y_dim] * len(units)
                        else:
                            self._subplot_y_titles.extend([y_dim] * len(units))

                        if self._subplot_z_titles is None:
                            self._subplot_z_titles = units.copy()
                        else:
                            self._subplot_z_titles.extend(units)
                    else:
                        if self._subplot_y_titles is None:
                            self._subplot_y_titles = units.copy()
                        else:
                            self._subplot_y_titles.extend(units)

            return method(self, *args, **kwargs)

        return wrapper

    @property
    def fig(self):
        """
        The Plotly figure object representing the chart.
        """
        if self._fig is None:
            self._fig = make_subplots(
                rows=self.rows,
                cols=self.columns,
                subplot_titles=self._subplot_titles,
                **self._subplots_kwargs,
            )
        else:
            current_ann = self._fig.layout.annotations or []
            # If title count changed, rebuild layout but keep traces & layout props
            if len(current_ann) != len(self._subplot_titles):
                existing_traces = list(self._fig.data)
                old_layout = self._fig.layout.to_plotly_json()
                new_fig = make_subplots(
                    rows=self.rows,
                    cols=self.columns,
                    subplot_titles=self._subplot_titles,
                    **self._subplots_kwargs,
                )
                for key, val in old_layout.items():
                    if key == "annotations":
                        continue
                    new_fig.layout[key] = val
                for trace in existing_traces:
                    new_fig.add_trace(trace)
                self._fig = new_fig
        return self._fig

    @property
    def rows(self):
        """The number of rows in the chart grid."""
        if self._rows is None:
            self._rows = 1
        return self._rows

    @property
    def columns(self):
        """The number of columns in the chart grid."""
        if self._columns is None:
            self._columns = 1
        return self._columns

    def add_trace(self, *args, **kwargs):
        """
        Adds a trace to the chart at the appropriate location.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to `plotly.graph_objects.Figure.add_trace`.
        **kwargs : dict
            Keyword arguments passed to `plotly.graph_objects.Figure.add_trace`.
        """
        self.fig.add_trace(*args, **kwargs)

    @set_subplot_titles
    def line(self, *args, **kwargs):
        """
        Adds a line plot to the chart.

        Parameters
        ----------
        data : array-like or earthkit.data.FieldList
            The data to be plotted.
        *args : tuple
            Positional arguments passed to the line plot generation function.
        **kwargs : dict
            Additional options for customizing the line plot.

        Notes
        -----
        Line plots are added as individual traces to each subplot.
        Titles are inferred from data attributes if not provided.
        """
        traces = line.line(*args, **kwargs)
        for i, trace in enumerate(traces):
            if isinstance(trace, list):
                if self._fig is None:
                    self._rows = self._rows or len(traces)
                    self._columns = self._columns or 1
                for sub_trace in trace:
                    self.add_trace(sub_trace, row=i + 1, col=1)
            else:
                self.add_trace(trace)

    @set_subplot_titles
    def box(self, *args, **kwargs):
        """
        Generate a set of box plot traces based on the provided data and quantiles.

        Parameters
        ----------
        data : array-like or earthkit.data.FieldList
            The data to be plotted.

        *args : tuple
            Positional arguments passed to the plotly `go.Box` constructors.

        quantiles : list of float, optional
            A list of quantiles to calculate for the data. The default is
            [0.05, 0.25, 0.5, 0.75, 0.95]. Note that any number of quantiles
            can be provided, but the default is based on the standard five-point
            box plot.

        time_axis : int, optional
            The axis along which to calculate the quantiles. The default is 0.

        **kwargs : dict
            Additional keyword arguments passed to the `go.Box` constructor.

        Returns
        -------
        list of plotly.graph_objects.Box

        Notes
        -----
        - The width of the box plots is scaled based on the x-axis spacing.
        - Extra boxes are added for quantiles beyond the standard five-point box plot.
        - Hover information is included for quantile scatter points, showing the
          quantile value and percentage.
        """
        traces = box.box(*args, **kwargs)
        for i, trace in enumerate(traces):
            if isinstance(trace, list):
                if self._fig is None:
                    self._rows = self._rows or len(traces)
                    self._columns = self._columns or 1
                for sub_trace in trace:
                    if not isinstance(sub_trace, (list, tuple)):
                        sub_trace = [sub_trace]
                    for actual_trace in sub_trace:
                        self.add_trace(actual_trace, row=i + 1, col=1)
            else:
                self.add_trace(trace)

    @set_subplot_titles
    def heatmap(self, *args, **kwargs):
        """
        Add a heat-map subplot (wrapper around :pyfunc:`heatmap.heatmap`).

        Parameters
        ----------
        *args, **kwargs
            Passed straight through to :pyfunc:`heatmap.heatmap` (after
            removing *colorscale* and *levels*).
        colorscale : str | list[str], default ``"Viridis"``
            * Either a single recognised Plotly colourscale name - applied
              to every subplot/trace;
            * **or** a list whose length equals the number of heat-map rows,
              providing a different colourscale per row.
        levels : list[np.ndarray] | None, optional
            *Controls colour binning.*

            * Each element must be a 1-D array of monotonically increasing
              bounds ``[zmin, …, zmax]``.
            * If the list has length 1, its single array is reused for every
              trace; otherwise its length **must** match the number of rows.
            * When *levels* is given, the corresponding *colorscale* is
              discretised into ``len(bounds) - 1`` constant bands whose
              edges follow *bounds*.

        Notes
        -----
        * Default behaviour is unchanged: 2-D view (*x* = time, *y* = second
          dimension) with no aggregation.
        * Shares insertion and layout logic with :pyfunc:`box` and
          :pyfunc:`line`.
        """

        def _is_named_color(color_str: str) -> bool:
            """
            Return True if `color_str` is a valid Plotly colour specification
            (hex, rgb/rgba, hsl, hsv or named), False otherwise.
            """
            import plotly.graph_objects as go

            try:
                go.Figure(data=[go.Scatter(marker=dict(color=color_str))])
                return True
            except (ValueError, TypeError):
                return False

        def _is_named_colorscale(name: str) -> bool:
            """Return *True* if *name* is a Plotly-recognised colorscale."""
            import plotly.colors as pc

            try:
                pc.get_colorscale(name)
                return True
            except Exception:
                return False

        kwargs.pop("time_axis", None)
        colorscale = kwargs.pop("colorscale", "Viridis")
        levels = kwargs.pop("levels", None)
        base_axis = 0
        if self._fig is not None:
            base_axis = len(self._fig.data)
        # transpose the data to put the time dimension last
        ds = inputs.to_xarray(args[0])
        time_dim = times.guess_time_dim(ds)
        new_order = [dim for dim in ds.dims if dim != time_dim] + [time_dim]
        ds = ds.transpose(*new_order)
        args_list = list(args)
        args_list[0] = ds
        args = tuple(args_list)
        base_traces = heatmap.heatmap(*args, **kwargs)
        traces = base_traces if isinstance(base_traces[0], list) else [base_traces]

        n_rows = len(traces)
        if isinstance(colorscale, str):
            cs_list = [colorscale] * n_rows
        elif isinstance(colorscale, Sequence):
            if all(_is_named_colorscale(c) for c in colorscale):
                if len(colorscale) != n_rows:
                    raise ValueError(
                        f"`colorscale` list length ({len(colorscale)}) "
                        f"must match number of rows ({n_rows})."
                    )
                cs_list = list(colorscale)
            elif all(isinstance(c, str) and _is_named_color(c) for c in colorscale):
                cs_list = [colorscale] * n_rows
            elif all(isinstance(c, Sequence) for c in colorscale):
                if len(colorscale) != n_rows:
                    raise ValueError(
                        f"`colorscale` list length ({len(colorscale)}) "
                        f"must match number of rows ({n_rows})."
                    )
                if all(all(_is_named_color(c) for c in cs) for cs in colorscale):
                    cs_list = colorscale
                else:
                    raise ValueError(
                        "`colorscale` list must contain either "
                        "named colours or named colorscales."
                    )
            else:
                raise TypeError("`colorscale` must be a string or list of strings.")
        else:
            raise TypeError("`colorscale` must be a string or list of strings.")

        if levels is None:
            lev_list = [None] * n_rows
        elif isinstance(levels, (Sequence, np.ndarray)):
            if len(levels) == 1:
                lev_list = list(levels) * n_rows
            elif not isinstance(levels[0], (Sequence, np.ndarray)):
                lev_list = [levels] * n_rows
            elif len(levels) == n_rows:
                lev_list = list(levels)
            else:
                raise ValueError(
                    f"`levels` must have length 1 or {n_rows} (rows), "
                    f"got {len(levels)}."
                )
            lev_list = [
                np.array(inner) if inner is not None else None for inner in lev_list
            ]
            for arr in lev_list:
                if arr is None:
                    continue
                if arr.ndim != 1:
                    raise TypeError("Each element of `levels` must be 1-D list.")
                if arr.size < 2 or not np.all(np.diff(arr) > 0):
                    raise ValueError(
                        "`levels` arrays must be increasing and length ≥ 2."
                    )
        else:
            raise TypeError("`levels` must be a list/tuple of numpy arrays or None.")

        rows_needed = self._rows or n_rows
        for i, (trace, cs_name, bounds) in enumerate(
            zip(traces, cs_list, lev_list), start=1
        ):
            for subtrace in trace:
                coloraxis_name = f"coloraxis{i+base_axis}"
                if bounds is not None:
                    custom_cmap = discrete_scale(cs_name, bounds)
                    subtrace.update(
                        coloraxis=coloraxis_name,
                        colorscale=custom_cmap,
                        zmin=bounds[0],
                        zmax=bounds[-1],
                    )
                else:
                    subtrace.update(coloraxis=coloraxis_name, colorscale=cs_name)

                if self._fig is None:
                    self._rows = rows_needed
                    self._columns = self._columns or 1

                y0, y1 = self.fig.layout[f"yaxis{i+base_axis}"].domain
                height = y1 - y0

                self._fig.update_layout(
                    **{
                        coloraxis_name: dict(
                            colorscale=subtrace.colorscale,
                            cmin=subtrace.zmin if hasattr(subtrace, "zmin") else None,
                            cmax=subtrace.zmax if hasattr(subtrace, "zmax") else None,
                            colorbar=dict(
                                y=y0,
                                yanchor="bottom",
                                len=height,
                                lenmode="fraction",
                                ypad=0,
                                tickvals=bounds,
                            ),
                        ),
                    }
                )
                self._fig.update_xaxes(
                    showspikes=True,
                    spikethickness=0,
                    spikemode="across",
                    spikesnap="cursor",
                )
                self.add_trace(subtrace, row=i + base_axis, col=1)

    @set_subplot_titles
    def bar(self, *args, **kwargs):
        """
        Adds a bar plot to the chart.

        Parameters
        ----------
        data : array-like or earthkit.data.FieldList
            The data to be plotted.
        *args : tuple
            Positional arguments passed to the bar plot generation function.
        **kwargs : dict
            Additional options for customizing the bar plot.

        Notes
        -----
        Bar plots are added as individual traces to each subplot.
        Titles are inferred from data attributes if not provided.
        """
        traces = bar.bar(*args, **kwargs)
        for i, trace in enumerate(traces):
            if isinstance(trace, list):
                if self._fig is None:
                    self._rows = self._rows or len(traces)
                    self._columns = self._columns or 1
                for sub_trace in trace:
                    self.add_trace(sub_trace, row=i + 1, col=1)
            else:
                self.add_trace(trace)

    def title(self, title):
        """
        Set the overall chart title.

        Parameters
        ----------
        title : str
            The title to display at the top of the chart.
        """
        self._layout_override["title"] = title

    def show(self, *args, **kwargs):
        """
        Display the chart.

        Parameters
        ----------
        *args : tuple
            Additional arguments for `plotly.graph_objects.Figure.show`.
        renderer : str, optional
            The renderer to use for displaying the chart. The default is "browser".
            For static plots, use "png".
        **kwargs : dict
            Additional options for rendering the chart.

        Returns
        -------
        None
        """
        layout = {
            **DEFAULT_LAYOUT,
            **self._layout_override,
        }
        # Temporary fix to remove _parent keys from nested dictionaries
        for k in layout:
            if isinstance(layout[k], dict):
                layout[k] = {
                    k2: v for k2, v in layout[k].items() if not k2.startswith("_")
                }
        self.fig.update_layout(**layout)
        for i in range(self.rows * self.columns):
            y_key = f"yaxis{i+1 if i>0 else ''}"
            x_key = f"xaxis{i+1 if i>0 else ''}"
            if self._subplot_x_titles:
                self.fig.update_layout(
                    **{
                        y_key: layout["yaxis"],
                        x_key: {
                            **layout["xaxis"],
                            **{"title": self._subplot_x_titles[i]},
                        },
                    }
                )
            if self._subplot_y_titles:
                self.fig.update_layout(
                    **{
                        x_key: layout["xaxis"],
                        y_key: {
                            **layout["yaxis"],
                            **{"title": self._subplot_y_titles[i]},
                        },
                    }
                )
            if self._subplot_z_titles:
                for i, ztitle in enumerate(self._subplot_z_titles):
                    ca = f"coloraxis{i+1}"
                    if ca in self.fig.layout:
                        self.fig.update_layout(
                            **{
                                ca: dict(
                                    colorbar=dict(
                                        title=dict(
                                            text=ztitle,
                                            side="right",
                                        ),
                                    )
                                )
                            }
                        )

        return self.fig.show(*args, **kwargs)
