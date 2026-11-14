from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib.axes import Axes

from edelassim.visualization.day_site import RdGn_colormap, day_site_score_plot
from edelassim.visualization.static_files import get_analysis_station_scores, get_open_loop_station_scores

if __name__ == "__main__":
    xpids = [
        "assim_viirs_cloudcover055_wy2122_soda_dev",
        "assim_viirs_cloudcover055_wy2122_python_pf",
        "assim_viirs_cloudcover055_wy2122_sigma_obs_var",
    ]
    plot_labels = ["SODA", "Python", "Sigma var"]
    period = slice("2021-11", "2022-06")

    sd_bdclim_ol_scores, fsc_bdclim_ol_scores = get_open_loop_station_scores(period=period)

    for idx, xpid in enumerate(xpids):
        fig, axs = plt.subplots(nrows=2, figsize=(18, 10))
        fig.suptitle(plot_labels[idx])
        sd_assim_scores, fsc_assim_scores = get_analysis_station_scores(xpid=xpid, period=period)

        score = xr.DataArray(1 - sd_assim_scores.data_vars["crps"] / sd_bdclim_ol_scores.data_vars["crps"], name="crpss")
        day_site_score_plot(score_data=score, ax=axs[0])
        axs[0].set_title(f"SD - {axs[0].get_title()}")

        score = xr.DataArray(1 - fsc_assim_scores.data_vars["crps"] / fsc_bdclim_ol_scores.data_vars["crps"], name="crpss")
        day_site_score_plot(score_data=score, ax=axs[1])
        axs[1].set_title(f"FSC -- {axs[1].get_title()}")

    plt.show()
