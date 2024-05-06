"""Plot OneMod results."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so


def plot_results(
    data: pd.DataFrame,
    x: str,
    y_dots: list[str] = [],
    y_line: list[str] = [],
    dots_options: dict = {},
    line_options: dict = {},
    facet_options: dict = {},
    share_options: dict = {},
    fig_options: dict = {},
    yscale: str = "linear",
) -> plt.Figure:
    """Plot result from OneMod model.

    Parameters
    ----------
    data : pandas.DataFrame
        OneMod results.
    x : str
        Column name for x-axis.
    y_dots : list of str, optional
        List of column names for scatter plots.
    y_line: list of str, optional
        List of column names for line plots.

    Returns
    -------
    matplotlib.Figure
        Figure object.

    Other Parameters
    ----------------
    dots_options : dict, optional
        Arguments passed to `matplotlib.pyplot.scatter() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        Dictionary keys must correspond to column names in `y_dots`.
    line_options : dict, optional
        Arguments passed to `matplotlib.pyplot.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.
        Dictionary keys must correspond to column names in `y_line`.
    facet_options : dict, optional
        Arguments passed to `seaborn.objects.Plot.facet() <https://seaborn.pydata.org/generated/seaborn.objects.Plot.facet.html>`_.
    share_options : dict, optional
        Arguments passed to `seaborn.objects.Plot.share() <https://seaborn.pydata.org/generated/seaborn.objects.Plot.share.html>`_.
    fig_options : dict, optional
        Arguments passed to `matplotlib.figure.Figure() <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_.
    yscale : str, optional
        Argument passed to `matplotlib.axes.Axes.set_yscale <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html>`_. Default is 'linear'.


    Returns
    -------
    plt.Figure
        Figure object.

    Example
    -------
    >>> from onemod_diagnostics.figure import plot_result
    >>> data = ...  # load result data for a single location
    >>> # plot result as time series
    >>> fig = plot_result(
    ...     data_sel,
    ...     x="year_id",
    ...     y_dots=["obs_rate"],
    ...     y_line=["truth", "regmod_smooth"],
    ...     dots_options=dict(obs_rate=dict(color="grey")),
    ...     facet_options=dict(col="age_mid", wrap=6),
    ...     fig_options=dict(figsize=(18, 12)),
    ... )
    >>> fig
    >>> # plot result as age series
    >>> fig = plot_result(
    ...     data_sel,
    ...     x="age_mid",
    ...     y_dots=["obs_rate"],
    ...     y_line=["truth", "regmod_smooth"],
    ...     dots_options=dict(obs_rate=dict(color="grey")),
    ...     facet_options=dict(col="year_id", wrap=6),
    ...     fig_options=dict(figsize=(18, 12)),
    ... )
    >>> fig

    """
    # Initialize figure and subplots
    fig = plt.Figure(**fig_options)
    so.Plot(data, x=x).facet(**facet_options).share(**share_options).on(
        fig
    ).plot()
    axes = fig.get_axes()
    by = [
        facet_options.get(key)
        for key in ["col", "row"]
        if facet_options.get(key) is not None
    ]

    # Query data by subplot
    if by:
        values = pd.DataFrame(
            data=[ax.get_title().split(" | ") for ax in axes],
            columns=by,
        ).astype(dict(zip(by, data[by].dtypes.to_list())))
        data_list = []
        for value in values.itertuples(index=False, name=None):
            selection = " & ".join(
                [f"{k} == {repr(v)}" for k, v in zip(by, value)]
            )
            data_list.append(data.query(selection))
    else:
        data_list = [data]

    # Plot data
    for ax, df in zip(axes, data_list):
        for y in y_dots:
            ax.scatter(df[x], df[y], label=y, **dots_options.get(y, {}))
        for y in y_line:
            ax.plot(df[x], df[y], label=y, **line_options.get(y, {}))

    # plot posinf and neginf
    for ax, df in zip(axes, data_list):
        ax.set_yscale(yscale)

        ylim = ax.get_ylim()
        for y in y_dots:
            df = df.query(f"{y} in [-inf, inf]").reset_index(drop=True)
            if not df.empty:
                df[y] = df[y].clip(*ylim)
                ax.scatter(df[x], df[y], label=y, **dots_options.get(y, {}))
        ax.set_ylim(ylim)

    # Format legend
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(handles),
    )

    return fig
