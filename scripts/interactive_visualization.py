import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from matplotlib.widgets import Button, Slider
from pyproj import CRS

from edelassim.snowlines import valid_snow_cover_fraction_s2, valid_snow_cover_fraction_viirs_mf
from edelassim.visualization import (
    FSC_CMAP_SNOW_COVER,
    METEOFRANCE_NEW_CLASSES,
    S2_CLASSES,
    add_colorbar,
    plot_snowline_polarplot_from_semidistributed,
)

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    ################################ User inputs #############################################
    s2_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/s2"
    edelweiss_folder = (
        "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/all_members"
    )
    viirs_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/meteofrance/"
    forcing_folder = "/home/imperatoren/work/edelweiss_assimilation/forcing/grandesrousses250m/daily_forcing"
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"
    forest_mask_path = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/forest_mask/forest_mask_corine_grandesrousses_max.nc"
    glacier_mask_path = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"

    labels = ("Sentinel-2", "Edelweiss", "VIIRS")
    colors = ("black", "blue", "red")

    # Initial date
    current_date = datetime(2021, 11, 1)

    ########################### Visualization ##########################################################
    logger.info("Plotting")

    glacier_mask = xr.open_dataset(glacier_mask_path)
    forest_mask = xr.open_dataset(forest_mask_path).sel(band=1)
    mask = glacier_mask.data_vars["__xarray_dataarray_variable__"] + forest_mask["__xarray_dataarray_variable__"]
    mask = georef_netcdf_rioxarray(mask, crs=CRS.from_epsg(2154))
    snow_cover_s2 = xr.open_dataset(f"{s2_folder}/spatial.nc").data_vars["snow_cover_fraction"]
    mask_20m = mask.rio.reproject_match(snow_cover_s2)
    snow_cover_s2 = valid_snow_cover_fraction_s2(snow_cover_s2.where(1 - mask_20m))

    snow_cover_edel = (
        xr.open_dataset(f"{edelweiss_folder}/spatial.nc")
        .data_vars["snow_cover_fraction"]
        .sortby("y", ascending=False)
        .where(1 - mask)
    )
    snow_cover_viirs = valid_snow_cover_fraction_viirs_mf(
        xr.open_dataset(f"{viirs_folder}/spatial.nc").data_vars["snow_cover_fraction"].where(1 - mask)
    )
    snow_cover_data = (snow_cover_s2, snow_cover_edel, snow_cover_viirs)
    forcing = xr.open_dataset(f"{forcing_folder}/spatial.nc").sortby("y", ascending=False).where(1 - mask)

    snowline_s2 = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc")
    snowline_edel = xr.open_dataset(f"{edelweiss_folder}/snowline_paremetrization.nc")
    snowline_viirs = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc")
    snowline_data = (snowline_s2, snowline_edel, snowline_viirs)

    snow_rain_forcing = xr.open_dataset(f"{forcing_folder}/snowline_parametrization.nc")
    snow_depth_edel = (
        xr.open_dataset(f"{edelweiss_folder}/pro/PRO_GrandesRousses250m_2021080206_2022080106.nc")
        .data_vars["DSN_T_ISBA"]
        .mean(dim="member")
        .sortby("y", ascending=False)
        .where(1 - mask)
    )
    # Determine which days are clear for observations
    n_pixels_area = snow_cover_s2.sizes["x"] * snow_cover_s2.sizes["y"]
    n_data_pixel = snow_cover_s2.count(dim=("x", "y"))
    cloud_mask_s2 = snow_cover_s2 >= S2_CLASSES["clouds"][0]
    cloud_flag_s2 = (cloud_mask_s2.sum(dim=("x", "y")) / n_data_pixel > 0.8) | (n_data_pixel / n_pixels_area < 0.3)
    good_dates_s2 = cloud_flag_s2.where(cloud_flag_s2 == 0, drop=True).time.values
    good_dates_s2 = [date.astype("M8[ms]").astype("O") for date in good_dates_s2]

    n_data_pixel = snow_cover_viirs.count(dim=("x", "y"))
    # print(n_data_pixel)
    # print(n_data_pixel)
    # cloud_mask_viirs = np.isnan(snow_cover_viirs)
    cloud_flag_viirs = (n_data_pixel / (snow_cover_viirs.sizes["x"] * snow_cover_viirs.sizes["y"])) < 0.50
    good_dates_viirs = cloud_flag_viirs.where(cloud_flag_viirs == 0, drop=True).time.values
    good_dates_viirs = [date.astype("M8[ms]").astype("O") for date in good_dates_viirs]

    # Slider for observation operator
    a_values = snowline_edel.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_a = a_values[0]

    # Create figure and polar axes
    fig = plt.figure(figsize=(20, 10))
    ax_snowlines = fig.add_subplot(2, 5, 5, projection="polar")
    ax_viirs = fig.add_subplot(2, 5, 3)
    ax_s2 = fig.add_subplot(2, 5, 1)
    ax_edelweiss = fig.add_subplot(2, 5, 2)
    ax_diff = fig.add_subplot(2, 5, 4)
    ax_snow_depth = fig.add_subplot(2, 5, 7)
    ax_precip = fig.add_subplot(2, 5, 8)
    ax_phase = fig.add_subplot(2, 5, 9)
    ax_snow_rain_line = fig.add_subplot(2, 5, 10, projection="polar")
    # ax_temperature = fig.add_subplot(258)
    axs_snow_cover = [ax_s2, ax_edelweiss, ax_viirs]
    axs_all = [*axs_snow_cover, ax_diff, ax_precip, ax_snowlines, ax_precip, ax_phase, ax_snow_rain_line]
    # fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    # fig.subplots_adjust(bottom=0.35)  # Room for buttons
    date_text = fig.suptitle(str(current_date.date()), y=0.92)
    a_text = fig.text(s=f"a = {current_a}", y=0.65, x=0.30)

    # Create button axes
    button_width = 0.1
    button_height = 0.075
    button_y1 = 0.15
    button_y2 = 0.05

    ax_d_minus = plt.axes([0.2, button_y1, button_width, button_height])
    ax_d_plus = plt.axes([0.35, button_y1, button_width, button_height])
    ax_m_minus = plt.axes([0.5, button_y1, button_width, button_height])
    ax_m_plus = plt.axes([0.65, button_y1, button_width, button_height])
    ax_a_minus = plt.axes([0.30, 0.60, 0.015, 0.04])
    ax_a_plus = plt.axes([0.33, 0.60, 0.015, 0.04])

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
    fig.subplots_adjust(bottom=0.35)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.98)

    def update_plot():
        [ax.clear() for ax in axs_all]

        for snowline, snow_cover, ax_snow_cover, label, color in zip(
            snowline_data, snow_cover_data, axs_snow_cover, labels, colors
        ):
            # print(label)
            # print(current_date)
            # # print(snowline.coords["time"].values)
            # print(np.datetime64(current_date) in snowline.coords["time"].values)
            if np.datetime64(current_date) in snowline.coords["time"]:
                snowline_to_plot = snowline.sel(time=current_date).sel(slope="10 - 30")
                if label == "Edelweiss":
                    snowline_to_plot = snowline_to_plot.sel(a=current_a)
                    snow_cover = snow_cover.sel(a=current_a)
            else:
                continue
            plot_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=snowline_to_plot,
                dataset_type="snow_cover",
                ax=ax_snowlines,
                label=label,
                color=color,
            )
            # print(axs[i])
            # FSC_CMAP_SNOW_COVER.set_bad("gray")
            FSC_CMAP_SNOW_COVER.set_bad("gray")
            ax_snow_cover.imshow(snow_cover.sel(time=current_date), cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
            ax_snow_cover.set_title(label)
            ax_snow_cover.set_xticks([]), ax_snow_cover.set_yticks([])

            add_colorbar(ax=ax_snow_cover)

        diff_edel_viirs = snow_cover_edel.sel(time=current_date, a=current_a) - snow_cover_viirs.sel(time=current_date)
        diff_cmap = plt.get_cmap("coolwarm_r")
        diff_cmap.set_bad("gray")
        ax_diff.imshow(diff_edel_viirs, cmap=diff_cmap, vmin=-1, vmax=1)
        ax_diff.set_title("Diff Edelweiss - VIIRS")
        ax_diff.set_xticks([]), ax_diff.set_yticks([])
        add_colorbar(ax=ax_diff)
        # [ax.legend() for ax in axs]

        ax_snow_depth.imshow(snow_depth_edel.sel(time=current_date), cmap="Blues_r")
        ax_snow_depth.set_title("Snow depth Edelweiss")
        ax_snow_depth.set_xticks([]), ax_snow_depth.set_yticks([])
        add_colorbar(ax=ax_snow_depth)

        total_precip = (forcing.data_vars["total_rain"] + forcing.data_vars["total_snow"]).sel(time=current_date) * 3600
        ax_precip.imshow(total_precip, cmap="viridis")  # , vmin=0, vmax=50)
        ax_precip.set_title("Total precipitation")
        ax_precip.set_xticks([]), ax_precip.set_yticks([])
        add_colorbar(ax=ax_precip)

        ax_phase.imshow(forcing.data_vars["phase_mean"].sel(time=current_date), cmap="Blues_r")
        ax_phase.set_title("Mean phase")
        ax_phase.set_xticks([]), ax_phase.set_yticks([])
        add_colorbar(ax=ax_phase)

        current_phase_line = snow_rain_forcing.sel(time=current_date)

        try:
            plot_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=current_phase_line,
                ax=ax_snow_rain_line,
                dataset_type="forcing",
            )
            ax_snow_rain_line.set_title("Forcing snow-rain line")
        except IndexError:
            print("ciao")

        date_text.set_text(str(current_date.date()))
        # Do not remove ticks for the snowline plot
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

    # plt.tight_layout()
    plt.show()
