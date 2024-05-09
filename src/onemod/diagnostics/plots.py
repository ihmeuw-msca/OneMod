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
    """Plot OneMod predictions for a single location.

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

    Examples
    -------
    Load OneMod results and filter by location.
    >>> import pandas as pd
    >>> from onemod.diagnostics import plot_results
    >>> data = pd.read_parquet(/path/to/dataframe).query("location_id == 71")

    Plot time series by sex and age.
    >>> from onemod.diagnostics import plot_result
    >>> fig = plot_results(
    >>>     data=data.query("30 <= age_mid <= 40"),
    >>>     x="year_id",
    >>>     y_dots=["obs_rate"],
    >>>     y_line=["true_rate", "pred_rate"],
    >>>     dots_options={"obs_rate": {"color": "gray"}},
    >>>     facet_options={"col": "sex_id", "row": "age_mid"},
    >>>     share_options={"y": False},
    >>>     fig_options={"figsize": (12, 8)},
    >>> )
    >>> fig.suptitle("death rate by sex and age", fontsize=16)
    >>> fig.tight_layout()
    >>> fig.savefig("example1.png", bbox_inches="tight")

    Plot age series by sex and year.
    >>>     fig = plot_results(
    >>>         data=pd.concat([
    >>>             data.query("sex_id == 1").rename(columns={
    >>>                 "obs_rate": "obs_male", "pred_rate": "pred_male"
    >>>             }),
    >>>             data.query("sex_id == 2").rename(columns={
    >>>                 "obs_rate": "obs_female", "pred_rate": "pred_female"
    >>>             }),
    >>>         ]).query("1981 <= year_id <= 1989"),
    >>>         x="age_mid",
    >>>         y_dots=["obs_male", "obs_female"],
    >>>         y_line=["pred_male", "pred_female"],
    >>>         dots_options={
    >>>             "obs_male": {"color": "gray", "marker": "o"},
    >>>             "obs_female": {"color": "gray", "marker": "^"},
    >>>         },
    >>>         facet_options={"col": "year_id", "wrap": 3},
    >>>         fig_options={"figsize": (18, 12)},
    >>>     )
    >>>     for ii, ax in enumerate(fig.get_axes()):
    >>>         if ii % 3 == 0:
    >>>             ax.set_ylabel("death_rate")
    >>>     fig.tight_layout()
    >>>     fig.savefig("example2.png", bbox_inches="tight")

    """
    # TODO: groupby within plot (i.e., if dim not a facet)
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

    # rescale and plot posinf/neginf
    # TODO: include infs after log/logit scale changes
    for ax, df in zip(axes, data_list):
        ax.set_yscale(yscale)
        ylim = ax.get_ylim()
        for y in y_dots:
            df_inf = df.query(f"{y} in [-inf, inf]").reset_index(drop=True)
            if not df_inf.empty:
                df_inf[y] = df_inf[y].clip(*ylim)
                ax.scatter(df_inf[x], df_inf[y], **dots_options.get(y, {}))
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
