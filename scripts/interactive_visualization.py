import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Button, Slider

from edelassim.snowlines import valid_snow_cover_fraction_viirs_mf
from edelassim.visualization import (
    FSC_CMAP_EDEL,
    FSC_CMAP_S2,
    FSC_CMAP_VIIRS_MF,
    METEOFRANCE_NEW_CLASSES,
    S2_CLASSES,
    add_colorbar,
    plot_snowline_polarplot,
)

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
    forest_mask_path = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/forest_mask/forest_mask_corine_grandesrousses_max.nc"
    glacier_mask_path = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"

    labels = ("Sentinel-2", "Edelweiss", "VIIRS")
    colors = ("black", "blue", "red")
    colormaps = (FSC_CMAP_S2, FSC_CMAP_EDEL, FSC_CMAP_VIIRS_MF)

    # Initial date
    current_date = datetime(2021, 11, 1)

    ########################### Visualization ##########################################################
    logger.info("Plotting")

    glacier_mask = xr.open_dataset(glacier_mask_path)
    forest_mask = xr.open_dataset(forest_mask_path).sel(band=1)
    mask = glacier_mask.data_vars["__xarray_dataarray_variable__"] + forest_mask["__xarray_dataarray_variable__"]

    snowline_s2 = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc")
    snowline_edel = xr.open_dataset(f"{edelweiss_folder}/snowline_paremetrization.nc")
    snowline_viirs = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc")
    snowline_data = (snowline_s2, snowline_edel, snowline_viirs)

    snow_cover_s2 = xr.open_dataset(f"{s2_folder}/regridded.nc").data_vars["snow_cover_fraction"]
    snow_cover_edel = (
        xr.open_dataset(f"{edelweiss_folder}/regridded.nc")
        .data_vars["snow_cover_fraction"]
        .sortby("y", ascending=False)
        .where(1 - mask)
    )
    snow_cover_viirs = xr.open_dataset(f"{viirs_folder}/regridded.nc").data_vars["snow_cover_fraction"].where(1 - mask)

    snow_cover_data = (snow_cover_s2, snow_cover_edel, snow_cover_viirs)

    # Determine which days are clear for observations
    n_pixels_area = snow_cover_s2.sizes["x"] * snow_cover_s2.sizes["y"]
    n_data_pixel = snow_cover_s2.count(dim=("x", "y"))
    cloud_mask_s2 = snow_cover_s2 == S2_CLASSES["clouds"][0]
    cloud_flag_s2 = (cloud_mask_s2.sum(dim=("x", "y")) / n_data_pixel > 0.8) | (n_data_pixel / n_pixels_area < 0.3)
    good_dates_s2 = cloud_flag_s2.where(cloud_flag_s2 == 0, drop=True).time.values
    good_dates_s2 = [date.astype("M8[ms]").astype("O") for date in good_dates_s2]

    n_data_pixel = snow_cover_viirs.count(dim=("x", "y"))
    cloud_mask_viirs = snow_cover_viirs == METEOFRANCE_NEW_CLASSES["clouds"][0]
    cloud_flag_viirs = (cloud_mask_viirs.sum(dim=("x", "y")) / n_data_pixel) > 0.35
    good_dates_viirs = cloud_flag_viirs.where(cloud_flag_viirs == 0, drop=True).time.values
    good_dates_viirs = [date.astype("M8[ms]").astype("O") for date in good_dates_viirs]

    # Slider for observation operator
    a_values = snowline_edel.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_a = a_values[0]

    # Create figure and polar axes
    fig = plt.figure(figsize=(20, 5))
    ax_snowlines = fig.add_subplot(155, projection="polar")
    ax_viirs = fig.add_subplot(153)
    ax_s2 = fig.add_subplot(151)
    ax_edelweiss = fig.add_subplot(152)
    ax_diff = fig.add_subplot(154)
    axs = [ax_s2, ax_edelweiss, ax_viirs, ax_diff, ax_snowlines]
    # fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    fig.subplots_adjust(bottom=0.25)  # Room for buttons
    date_text = fig.suptitle(str(current_date.date()), y=0.92)
    a_text = fig.text(s=f"a = {current_a}", y=0.80, x=0.285)

    # Create button axes
    button_width = 0.1
    button_height = 0.075
    button_y1 = 0.15
    button_y2 = 0.05

    ax_d_minus = plt.axes([0.2, button_y1, button_width, button_height])
    ax_d_plus = plt.axes([0.35, button_y1, button_width, button_height])
    ax_m_minus = plt.axes([0.5, button_y1, button_width, button_height])
    ax_m_plus = plt.axes([0.65, button_y1, button_width, button_height])
    ax_a_minus = plt.axes([0.27, 0.75, 0.015, 0.04])
    ax_a_plus = plt.axes([0.30, 0.75, 0.015, 0.04])

    ax_next_good_viirs = plt.axes([0.5, button_y2, button_width, button_height])
    ax_prev_good_viirs = plt.axes([0.35, button_y2, button_width, button_height])
    ax_next_good_s2 = plt.axes([0.65, button_y2, button_width, button_height])
    ax_prev_good_s2 = plt.axes([0.2, button_y2, button_width, button_height])

    btn_d_minus = Button(ax_d_minus, "D-")
    btn_d_plus = Button(ax_d_plus, "D+")
    btn_m_minus = Button(ax_m_minus, "M-")
    btn_m_plus = Button(ax_m_plus, "M+")
    btn_next_good_viirs = Button(ax_next_good_viirs, "Next good VIIRS")
    btn_prev_good_viirs = Button(ax_prev_good_viirs, "Prev Good VIIRS")
    btn_next_good_s2 = Button(ax_next_good_s2, "Next good S2")
    btn_prev_good_s2 = Button(ax_prev_good_s2, "Prev Good S2")
    btn_a_minus = Button(ax_a_minus, "a-")
    btn_a_plus = Button(ax_a_plus, "a+")
    plt.subplots_adjust(bottom=-0.25)

    def update_plot():
        [ax.clear() for ax in axs]

        for i, (snowline, snow_cover, label, color, colormap) in enumerate(
            zip(snowline_data, snow_cover_data, labels, colors, colormaps)
        ):
            try:
                snowline_to_plot = snowline.sel(time=current_date).sel(slope="10 - 30")
                if label == "Edelweiss":
                    snowline_to_plot = snowline_to_plot.sel(a=current_a)
                    snow_cover = snow_cover.sel(a=current_a)
                plot_snowline_polarplot(
                    snowline_parametrization_dataset=snowline_to_plot,
                    ax=ax_snowlines,
                    label=label,
                    color=color,
                )

                axs[i].imshow(snow_cover.sel(time=current_date), cmap=colormap)
                axs[i].set_title(label)
                # Do not remove ticks for the snowline plot
                [ax.set_xticks([]) for ax in axs[:-1]]
                [ax.set_yticks([]) for ax in axs[:-1]]
                add_colorbar(ax=axs[i])
                # fig.colorbar()
            except KeyError:
                continue

        diff_edel_viirs = snow_cover_edel.sel(time=current_date, a=current_a) - valid_snow_cover_fraction_viirs_mf(
            snow_cover_viirs.sel(time=current_date)
        )

        im_diff = ax_diff.imshow(diff_edel_viirs, cmap="coolwarm_r", vmin=-1, vmax=1)
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

    def next_good_date_viirs(event):
        global current_date

        # Find the first good date after current_date
        for d in good_dates_viirs:
            if d > current_date:
                current_date = d
                break
        update_plot()

    def prev_good_date_viirs(event):
        global current_date

        # Find the last good date before current_date
        for d in reversed(good_dates_viirs):
            if d < current_date:
                current_date = d
                break

        update_plot()

    def next_good_date_s2(event):
        global current_date

        # Find the first good date after current_date
        for d in good_dates_s2:
            if d > current_date:
                current_date = d
                break
        update_plot()

    def prev_good_date_s2(event):
        global current_date

        # Find the last good date before current_date
        for d in reversed(good_dates_s2):
            if d < current_date:
                current_date = d
                break

        update_plot()

    # def update_a(val):
    #     global current_a
    #     # Find closest value
    #     current_a = min(a_values, key=lambda x: abs(x - val))
    #     slider_a.set_val(current_a)  # Snap slider
    #     update_plot()

    def change_a(delta):
        global current_a
        current_a_idx = list(a_values).index(current_a)
        current_a_idx += delta
        current_a = a_values[current_a_idx]
        a_text.set_text(f"a = {current_a}")
        update_plot()

    btn_d_minus.on_clicked(lambda e: change_day(-1))
    btn_d_plus.on_clicked(lambda e: change_day(1))
    btn_m_minus.on_clicked(lambda e: change_month(-1))
    btn_m_plus.on_clicked(lambda e: change_month(1))

    btn_next_good_viirs.on_clicked(next_good_date_viirs)
    btn_prev_good_viirs.on_clicked(prev_good_date_viirs)
    btn_next_good_s2.on_clicked(next_good_date_s2)
    btn_prev_good_s2.on_clicked(prev_good_date_s2)

    btn_a_minus.on_clicked(lambda e: change_a(-1))
    btn_a_plus.on_clicked(lambda e: change_a(1))
    # Initial plot

    update_plot()

    plt.tight_layout()
    plt.show()
