import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from matplotlib.widgets import Button, Slider
from pyproj import CRS

from edelassim.observation_operators import dickinson
from edelassim.snowlines import valid_snow_cover_fraction_s2, valid_snow_cover_fraction_viirs_mf
from edelassim.visualization import (
    FSC_CMAP_SNOW_COVER,
    METEOFRANCE_NEW_CLASSES,
    S2_CLASSES,
    add_colorbar,
    plot_elevation_lines,
    plot_ensemble_snowline_polarplot_from_semidistributed,
    plot_snowline_polarplot_from_semidistributed,
)


def quantile_index(arr, q):
    """Return the index of the value closest to the q-quantile."""
    q_val = np.quantile(arr, q)
    return np.argmin(np.abs(arr - q_val))


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


def change_mb(delta):
    global current_member
    current_mb_idx = list(member_values).index(current_member)
    current_mb_idx += delta
    current_member = member_values[current_mb_idx]
    mb_text.set_text(f"member = {current_member}")
    update_plot()


def update_plot():
    [ax.clear() for ax in axs_all]
    # Determine ensemble visualization indexes

    for snowline, label, color in zip(snowline_data, labels, colors):
        if np.datetime64(current_date) in snowline.coords["time"]:
            snowline_to_plot = snowline.sel(time=current_date, slope="8 - 30")
            if label == "Edelweiss":
                plot_ensemble_snowline_polarplot_from_semidistributed(
                    snowline_parametrization_dataset=snowline_to_plot.sel(a=current_a),
                    dataset_type="snow_cover",
                    ax=ax_snowlines,
                    color=color,
                )
                snowline_to_plot = snowline_to_plot.sel(a=current_a, member=current_member)
        else:
            continue

        plot_snowline_polarplot_from_semidistributed(
            snowline_parametrization_dataset=snowline_to_plot,
            dataset_type="snow_cover",
            ax=ax_snowlines,
            label=label,
            color=color,
        )

    FSC_CMAP_SNOW_COVER.set_bad("gray")
    if np.datetime64(current_date) in snow_cover_s2.coords["time"]:
        ax_s2.imshow(snow_cover_s2.sel(time=current_date), cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
        add_colorbar(ax=ax_s2)
        plot_elevation_lines(ax=ax_s2, dem=dem_20m)
    ax_s2.set_title("Sentinel-2")
    ax_s2.set_xticks([]), ax_s2.set_yticks([])

    ax_viirs.imshow(snow_cover_viirs.sel(time=current_date), cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    ax_viirs.set_title("VIIRS")
    ax_viirs.set_xticks([]), ax_viirs.set_yticks([])
    add_colorbar(ax=ax_viirs)
    plot_elevation_lines(ax=ax_viirs, dem=dem_250m)

    fsc_edel = dickinson(sd=snow_depth_edel.sel(time=current_date, member=current_member), a=current_a, b=0.11)
    ax_edelweiss.imshow(fsc_edel, cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    ax_edelweiss.set_title("Edelweiss FSC")
    ax_edelweiss.set_xticks([]), ax_edelweiss.set_yticks([])
    add_colorbar(ax=ax_edelweiss)
    plot_elevation_lines(ax=ax_edelweiss, dem=dem_250m)

    diff_edel_viirs = fsc_edel - snow_cover_viirs.sel(time=current_date)
    diff_cmap = plt.get_cmap("coolwarm_r")
    diff_cmap.set_bad("gray")
    ax_diff.imshow(diff_edel_viirs, cmap=diff_cmap, vmin=-1, vmax=1)
    ax_diff.set_title("Diff Edelweiss - VIIRS")
    ax_diff.set_xticks([]), ax_diff.set_yticks([])
    add_colorbar(ax=ax_diff)
    plot_elevation_lines(ax=ax_diff, dem=dem_250m)

    snow_depth_cmap = plt.get_cmap("Blues")
    snow_depth_cmap.set_under("black")
    snow_depth_cmap.set_bad("gray")
    ax_snow_depth.imshow(snow_depth_edel.sel(time=current_date, member=current_member), cmap=snow_depth_cmap, vmin=0.001)
    ax_snow_depth.set_title("Snow depth Edelweiss")
    ax_snow_depth.set_xticks([]), ax_snow_depth.set_yticks([])
    add_colorbar(ax=ax_snow_depth)
    plot_elevation_lines(ax=ax_snow_depth, dem=dem_250m)

    total_precip = forcing.data_vars["precip_total"].sel(time=current_date, member=current_member)
    precip_cmap = plt.get_cmap("viridis")
    precip_cmap.set_bad("gray")
    ax_precip.imshow(total_precip, cmap=precip_cmap)  # , vmin=0, vmax=50)
    ax_precip.set_title("Total precipitation")
    ax_precip.set_xticks([]), ax_precip.set_yticks([])
    add_colorbar(ax=ax_precip)
    plot_elevation_lines(ax=ax_precip, dem=dem_250m)

    ######### FORCING ###########
    phase_cmap = plt.get_cmap("Blues_r")
    phase_cmap.set_bad("gray")
    ax_phase.imshow(forcing.data_vars["phase"].sel(time=current_date, member=current_member), cmap=phase_cmap, vmin=-1, vmax=1)
    ax_phase.set_title("Phase")
    ax_phase.set_xticks([]), ax_phase.set_yticks([])
    add_colorbar(ax=ax_phase)
    plot_elevation_lines(ax=ax_phase, dem=dem_250m)

    try:
        if not np.all(np.isnan(snow_rain_forcing.sel(time=current_date, slope_bins="8 - 30").data_vars["phase"])):
            plot_ensemble_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=snow_rain_forcing.sel(time=current_date, slope_bins="8 - 30"),
                ax=ax_snow_rain_line,
                dataset_type="forcing",
            )
            current_phase_line = snow_rain_forcing.sel(time=current_date, member=current_member, slope_bins="8 - 30")
            plot_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=current_phase_line,
                ax=ax_snow_rain_line,
                dataset_type="forcing",
            )
            ax_snow_rain_line.set_title("Forcing snow-rain line")
        else:
            logging.info(f"no phase on day {current_date}")
    except ValueError as e:
        logging.info(f"Exception caught {e}")

    date_text.set_text(str(current_date.date()))
    # Do not remove ticks for the snowline plot
    fig.canvas.draw_idle()


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
    dem_250m_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    dem_20m_filepath = f"{topography_data_folder}/20m/DEM_GR_UTM_20m.tif"
    labels = ("Sentinel-2", "Edelweiss", "VIIRS")
    colors = ("black", "blue", "red")

    # Initial date
    current_date = datetime(2021, 11, 1)

    ########################### Visualization ##########################################################
    logger.info("Plotting")

    glacier_mask = xr.open_dataset(glacier_mask_path).data_vars["__xarray_dataarray_variable__"]
    forest_mask = xr.open_dataset(forest_mask_path).sel(band=1).data_vars["__xarray_dataarray_variable__"]
    mask = glacier_mask + forest_mask
    mask = georef_netcdf_rioxarray(mask, crs=CRS.from_epsg(2154))
    snow_cover_s2 = xr.open_dataset(f"{s2_folder}/spatial.nc").data_vars["snow_cover_fraction"]
    mask_20m = mask.rio.reproject_match(snow_cover_s2)
    snow_cover_s2 = valid_snow_cover_fraction_s2(snow_cover_s2.where(1 - mask_20m))

    dem_250m = xr.open_dataarray(dem_250m_filepath).sel(band=1)
    dem_20m = xr.open_dataarray(dem_20m_filepath).sel(band=1)

    snow_cover_viirs = valid_snow_cover_fraction_viirs_mf(
        xr.open_dataset(f"{viirs_folder}/spatial.nc").data_vars["snow_cover_fraction"].where(1 - mask)
    )

    forcing = xr.open_mfdataset(f"{forcing_folder}/spatial.nc").sortby("y", ascending=False).where(1 - mask)

    snowline_s2 = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc")
    snowline_edel = xr.open_dataset(f"{edelweiss_folder}/snowline_paremetrization.nc")
    snowline_viirs = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc")
    snowline_data = (snowline_s2, snowline_edel, snowline_viirs)

    snow_rain_forcing = xr.open_dataset(f"{forcing_folder}/snowline_parametrization.nc")
    snow_depth_edel = (
        xr.open_dataset(f"{edelweiss_folder}/pro/PRO_GrandesRousses250m_2021080206_2022080106.nc")
        .data_vars["DSN_T_ISBA"]
        .sortby("y", ascending=False)
        .where(1 - mask)
    )

    # Determine which days are clear for observations
    n_pixels_area = snow_cover_s2.sizes["x"] * snow_cover_s2.sizes["y"]
    n_data_pixel = snow_cover_s2.count(dim=("x", "y"))
    cloud_mask_s2 = snow_cover_s2 >= S2_CLASSES["clouds"][0]
    # Quick criterium to determine whether a Sentinel-2 image is exploitable
    cloud_flag_s2 = (cloud_mask_s2.sum(dim=("x", "y")) / n_data_pixel > 0.8) | (n_data_pixel / n_pixels_area < 0.3)
    good_dates_s2 = cloud_flag_s2.where(cloud_flag_s2 == 0, drop=True).time.values
    good_dates_s2 = [date.astype("M8[ms]").astype("O") for date in good_dates_s2]

    n_data_pixel = snow_cover_viirs.count(dim=("x", "y"))

    # cloud_mask_viirs = np.isnan(snow_cover_viirs)
    cloud_flag_viirs = (n_data_pixel / (snow_cover_viirs.sizes["x"] * snow_cover_viirs.sizes["y"])) < 0.50
    good_dates_viirs = cloud_flag_viirs.where(cloud_flag_viirs == 0, drop=True).time.values
    good_dates_viirs = [date.astype("M8[ms]").astype("O") for date in good_dates_viirs]

    # Slider for observation operator
    a_values = snowline_edel.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_a = a_values[0]

    # Slider for member
    member_values = snowline_edel.coords["member"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_member = member_values[0]

    # Create figure and polar axes
    fig = plt.figure(figsize=(20, 10))

    ax_s2 = fig.add_subplot(3, 4, 1)
    ax_edelweiss = fig.add_subplot(3, 4, 2)
    ax_viirs = fig.add_subplot(3, 4, 3)
    ax_diff = fig.add_subplot(3, 4, 4)
    ax_snow_depth = fig.add_subplot(3, 4, 6)
    ax_precip = fig.add_subplot(3, 4, 7)
    ax_phase = fig.add_subplot(3, 4, 8)
    ax_snowlines = fig.add_subplot(3, 4, 11, projection="polar")
    ax_snow_rain_line = fig.add_subplot(3, 4, 12, projection="polar")
    # ax_temperature = fig.add_subplot(258)
    axs_snow_cover = [ax_s2, ax_edelweiss, ax_viirs]
    axs_all = [*axs_snow_cover, ax_diff, ax_precip, ax_snowlines, ax_snow_depth, ax_precip, ax_phase, ax_snow_rain_line]
    # fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    # fig.subplots_adjust(bottom=0.35)  # Room for buttons
    date_text = fig.suptitle(str(current_date.date()), y=0.98)
    a_text = fig.text(s=f"a = {current_a}", y=0.32, x=0.37)
    mb_text = fig.text(s=f"member = {current_member}", y=0.25, x=0.37)

    # Create button axes
    button_width = 0.07
    button_height = 0.025
    button_y1 = 0.12
    button_y2 = 0.08
    button_x1 = 0.02

    ax_d_minus = plt.axes([button_x1 + button_width, button_y1, button_width, button_height])
    ax_d_plus = plt.axes([button_x1 + 2 * button_width, button_y1, button_width, button_height])
    ax_m_minus = plt.axes([button_x1, button_y1, button_width, button_height])
    ax_m_plus = plt.axes([button_x1 + 3 * button_width, button_y1, button_width, button_height])
    ax_a_minus = plt.axes([0.35, 0.35, 0.03, 0.02])
    ax_a_plus = plt.axes([0.40, 0.35, 0.03, 0.02])
    ax_mb_minus = plt.axes([0.35, 0.20, 0.03, 0.02])
    ax_mb_plus = plt.axes([0.40, 0.20, 0.03, 0.02])

    ax_next_good_viirs = plt.axes([button_x1 + 2 * button_width, button_y2, button_width, button_height])
    ax_prev_good_viirs = plt.axes([button_x1 + button_width, button_y2, button_width, button_height])
    ax_next_good_s2 = plt.axes([button_x1 + 3 * button_width, button_y2, button_width, button_height])
    ax_prev_good_s2 = plt.axes([button_x1, button_y2, button_width, button_height])

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
    btn_mb_minus = Button(ax_mb_minus, "member-")
    btn_mb_plus = Button(ax_mb_plus, "member+")

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

    btn_mb_minus.on_clicked(lambda e: change_mb(-1))
    btn_mb_plus.on_clicked(lambda e: change_mb(1))

    # Initial plot
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(top=0.96)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.96)
    fig.subplots_adjust(wspace=0.1)
    fig.subplots_adjust(hspace=0.1)

    update_plot()

    # plt.tight_layout()
    plt.show()
