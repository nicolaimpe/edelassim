import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.widgets import Button

from edelassim.snowlines import valid_snow_cover_fraction_viirs_mf
from edelassim.visualization import FSC_CMAP_EDEL, FSC_CMAP_S2, FSC_CMAP_VIIRS_MF, plot_snowline_polarplot

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    ################################ User inputs #############################################
    s2_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/s2"
    edelweiss_folder = (
        "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/all_members"
    )
    viirs_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/meteofrance/"

    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/"

    labels = ("Sentinel-2", "Edelweiss", "VIIRS")
    colors = ("black", "blue", "red")
    colormaps = (FSC_CMAP_S2, FSC_CMAP_EDEL, FSC_CMAP_VIIRS_MF)

    # Initial date
    current_date = datetime(2021, 10, 1)
    ########################### Visualization ##########################################################
    logger.info("Plotting")

    snowline_s2 = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc")
    snowline_edel = xr.open_dataset(f"{edelweiss_folder}/snowline_paremetrization.nc")
    snowline_viirs = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc")
    snowline_data = (snowline_s2, snowline_edel, snowline_viirs)

    snow_cover_s2 = xr.open_dataset(f"{s2_folder}/regridded.nc").data_vars["snow_cover_fraction"]
    snow_cover_edel = xr.open_dataset(f"{edelweiss_folder}/regridded.nc", engine="rasterio").data_vars["snow_cover_fraction"]
    snow_cover_viirs = xr.open_dataset(f"{viirs_folder}/regridded.nc").data_vars["snow_cover_fraction"]

    snow_cover_data = (snow_cover_s2, snow_cover_edel, snow_cover_viirs)

    # cloud_flag_s2 = snow_cover_s2 == S2_CLASSES
    # Create figure and polar axes
    fig = plt.figure(figsize=(15, 10))
    ax_snowlines = fig.add_subplot(155, projection="polar")
    ax_viirs = fig.add_subplot(153)
    ax_s2 = fig.add_subplot(151)
    ax_edelweiss = fig.add_subplot(152)
    ax_diff = fig.add_subplot(154)
    axs = [ax_s2, ax_edelweiss, ax_viirs, ax_diff, ax_snowlines]
    # fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    fig.subplots_adjust(bottom=0.25)  # Room for buttons

    # Create button axes
    button_width = 0.1
    button_height = 0.075
    button_y = 0.05

    ax_d_minus = plt.axes([0.2, button_y, button_width, button_height])
    ax_d_plus = plt.axes([0.35, button_y, button_width, button_height])
    ax_m_minus = plt.axes([0.5, button_y, button_width, button_height])
    ax_m_plus = plt.axes([0.65, button_y, button_width, button_height])

    btn_d_minus = Button(ax_d_minus, "D-")
    btn_d_plus = Button(ax_d_plus, "D+")
    btn_m_minus = Button(ax_m_minus, "M-")
    btn_m_plus = Button(ax_m_plus, "M+")

    def update_plot():
        [ax.clear() for ax in axs]

        date_text = fig.suptitle(str(current_date.date()), y=0.92)
        for i, (snowline, snow_cover, label, color, colormap) in enumerate(
            zip(snowline_data, snow_cover_data, labels, colors, colormaps)
        ):
            try:
                plot_data = snowline.sel(time=current_date).sel(slope="10 - 30")
                plot_snowline_polarplot(
                    snowline_parametrization_dataset=plot_data,
                    ax=ax_snowlines,
                    label=label,
                    color=color,
                )

                axs[i].imshow(snow_cover.sel(time=current_date), cmap=colormap)
                axs[i].set_title(label)
            except KeyError:
                continue

        diff_edel_viirs = snow_cover_edel.sel(time=current_date) - valid_snow_cover_fraction_viirs_mf(
            snow_cover_viirs.sel(time=current_date)
        )
        ax_diff.imshow(diff_edel_viirs, cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax_diff.set_title("Diff Edelweiss - VIIRS")
        # [ax.legend() for ax in axs]

        date_text.set_text(str(current_date.date()))
        fig.canvas.draw_idle()

    def change_day(delta):
        global current_date
        current_date += timedelta(days=delta)
        update_plot()

    def change_month(delta):
        global current_date
        year = current_date.year
        month = current_date.month + delta
        if month > 12:
            month = 1
            year += 1
        elif month < 1:
            month = 12
            year -= 1
        # Handle day overflow (e.g., Jan 31 -> Feb)
        last_day = calendar.monthrange(year, month)[1]
        day = min(current_date.day, last_day)
        current_date = datetime(year, month, day)
        update_plot()

    btn_d_minus.on_clicked(lambda e: change_day(-1))
    btn_d_plus.on_clicked(lambda e: change_day(1))
    btn_m_minus.on_clicked(lambda e: change_month(-1))
    btn_m_plus.on_clicked(lambda e: change_month(1))

    # Initial plot

    update_plot()
    # Do not remove ticks for the snowline plot
    [ax.set_xticks([]) for ax in axs[:-1]]
    [ax.set_yticks([]) for ax in axs[:-1]]
    plt.tight_layout()
    plt.show()
