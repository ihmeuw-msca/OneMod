"""Plot OneMod results."""

from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from loguru import logger
from pplkit.data.interface import DataInterface


def plot_results(
    data: pd.DataFrame,
    x: str,
    y_dots: list[str] = [],
    y_line: list[str] = [],
    y_fill: list[str] = [],
    dots_options: dict = {},
    line_options: dict = {},
    fill_options: dict = {},
    facet_options: dict = {},
    share_options: dict = {},
    fig_options: dict = {},
    legend_options: dict = {},
    yscale: str = "linear",
    fig: plt.Figure | None = None,
    legend: bool = True,
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
    y_fill: list of tuple of str, optional
        List of column names for uncertainty intervals. For each column
        in `y_fill`, e.g. 'pred', data must contain column lower and
        upper bounds, e.g. 'pred_lwr' and 'pred_upr'.

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
    fill_options : dict, optional
        Arguments passed to `matplotlib.pyplot.fill_between() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html>`_.
        Dictionary keys must correspond to column names in `y_fill`.
    facet_options : dict, optional
        Arguments passed to `seaborn.objects.Plot.facet() <https://seaborn.pydata.org/generated/seaborn.objects.Plot.facet.html>`_.
    share_options : dict, optional
        Arguments passed to `seaborn.objects.Plot.share() <https://seaborn.pydata.org/generated/seaborn.objects.Plot.share.html>`_.
    fig_options : dict, optional
        Arguments passed to `matplotlib.figure.Figure() <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_.
    legend_options : dict, optional
        Arguments passed to `matplotlib.figure.Figure.legend() <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.legend.html#matplotlib.figure.Figure.legend>`_.
    yscale : str, optional
        Argument passed to `matplotlib.axes.Axes.set_yscale <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html>`_. Default is 'linear'.
    fig : matplotlib.Figure, optional
        Existing figure to add plots to. If None, create new figure.
        Default is None.
    legend : bool, optional
        Whether to display plot legend. Default is True.

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
    if fig is None:
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
            data=[ax.get_title().split(" | ") for ax in axes], columns=by
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
        for y in y_fill:
            ax.fill_between(
                df[x],
                df[f"{y}_lwr"],
                df[f"{y}_upr"],
                label=y,
                **fill_options.get(y, {}),
            )
        for y in y_line:
            ax.plot(df[x], df[y], label=y, **line_options.get(y, {}))
        for y in y_dots:
            ax.scatter(df[x], df[y], label=y, **dots_options.get(y, {}))

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
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        default_options = dict(
            loc="lower_center", bbox_to_anchor=(0.5, -0.05), ncol=len(handles)
        )
        if legend_options:
            default_options.update(legend_options)
        fig.legend(handles, labels, **default_options)

    return fig


def plot_rover_covsel_results(
    summaries: pd.DataFrame, covs: list[str] | None = None
) -> plt.Figure:
    """Description.

    Parameters
    ----------
    summaries : pd.DataFrame
        Description.
    covs : list[str]
        Description.

    Returns
    -------
    matplotlib.Figure
        Figure object.

    """
    logger.info("Plotting coefficient magnitudes by age.")

    # add age_mid to summary
    df_covs = summaries.groupby("cov")
    covs = covs or list(df_covs.groups.keys())
    logger.info(
        f"Starting to plot for {len(covs)} covariates and {df_covs['age_group_id'].nunique()} age groups"
    )

    fig, ax = plt.subplots(len(covs), 1, figsize=(8, 2 * len(covs)))
    ax = [ax] if len(covs) == 1 else ax
    for ii, cov in enumerate(covs):
        df_cov = df_covs.get_group(cov).sort_values(by="age_mid")
        if ii % 5 == 0:
            logger.info(f"Plotting for group {ii}")
        ax[ii].errorbar(
            df_cov["age_mid"],
            df_cov["coef"],
            yerr=1.96 * df_cov["coef_sd"],
            fmt="o-",
            alpha=0.5,
            label="rover_covsel",
        )
        ax[ii].set_ylabel("cov")
        ax[ii].axhline(0.0, linestyle="--")

    logger.info("Completed plotting of rover results.")
    return fig


def plot_spxmod_results(
    dataif: DataInterface, summaries: pd.DataFrame
) -> plt.Figure | None:
    """Description.

    Parameters
    ----------
    dataif : DataInterface
        Description.
    summaries : pandas.DataFrame
        Description.

    Returns
    -------
    matplotlib.Figure
        Figure object.

    """
    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    if not selected_covs:
        warn("No covariates selected; skipping `plot_spxmod_results`")
        return None

    df_covs = dataif.load_spxmod("coef.csv").groupby("cov")

    fig = plot_rover_covsel_results(dataif, summaries, covs=selected_covs)
    logger.info(
        f"Plotting smoothed covariates for {len(selected_covs)} covariates"
    )
    for ax, cov in zip(fig.axes, selected_covs):
        df_cov = df_covs.get_group(cov)
        ax.plot(
            df_cov["age_mid"], df_cov["coef"], "o-", alpha=0.5, label="spxmod"
        )
        ax.legend(fontsize="xx-small")
    return fig
