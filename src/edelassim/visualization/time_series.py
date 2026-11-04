import xarray as xr
from matplotlib.axes import Axes


def plot_ensemble_time_series(time_series_data: xr.DataArray, mb_to_plot: int, ax: Axes, color: str, label: str):

    t_coord = time_series_data.coords["time"]
    qmin = time_series_data.quantile(q=0.1, dim="member")
    qmax = time_series_data.quantile(q=0.9, dim="member")
    mb_data = time_series_data.sel(member=mb_to_plot)
    ax.fill_between(t_coord, qmin, qmax, color=color, alpha=0.2)
    ax.plot(t_coord, mb_data, color=color, linestyle="dashed", linewidth=3, label=label)
    for mb in time_series_data.coords["member"]:
        ax.plot(t_coord, time_series_data.sel(member=mb), linestyle="dotted", color=color, linewidth=0.6)


def add_assim_dates(assim_dates, ax: Axes):
    ax.vlines(
        assim_dates,
        ymin=ax.get_ybound()[0],
        ymax=ax.get_ybound()[1],
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
        label="Assimilation dates",
    )
