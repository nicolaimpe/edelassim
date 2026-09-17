import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from pyproj import CRS, Proj, Transformer, transform
from sklearn.metrics import mean_squared_error

from edelassim.bdclim import find_station_imshow_location, find_station_locations
from edelassim.evaluations import compute_rmse
from edelassim.observation_operators import dickinson
from edelassim.observations import (
    find_clear_dates_s2,
    find_clear_dates_viirs,
    valid_snow_cover_fraction_s2,
    valid_snow_cover_fraction_viirs_mf,
)
from edelassim.snowlines import find_forcing_snowrain_line, find_snowline_from_snow_penalization
from edelassim.visualization.interactive_plots import InteractiveSeasonExploreButtons
from edelassim.visualization.polar import plot_polar_envelop_member, plot_snowline_polarplot, set_polarplot
from edelassim.visualization.spatial import (
    FIELD_DIFF_CMAP,
    FSC_CMAP_SNOW_COVER,
    PHASE_CMAP,
    PRECIP_CMAP,
    SNOW_DEPTH_CMAP,
    add_2d_plot,
)
from edelassim.visualization.time_series import plot_ensemble_time_series


def quantile_index(arr: np.ndarray, q: float):
    """Return the index of the value closest to the q-quantile."""
    q_val = np.quantile(arr, q)
    return np.argmin(np.abs(arr - q_val))


def change_day(delta: int):
    global current_date
    current_date += timedelta(days=delta)
    update_all_plots()


def change_month(delta: int):
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
    update_all_plots()


def last_good_date(good_dates_list: list[datetime]):
    global current_date
    # Find the last good date before current_date
    for d in reversed(good_dates_list):
        if d < current_date:
            current_date = d
            break
    update_all_plots()


def next_good_date(good_dates_list: list[datetime]):
    global current_date

    # Find the first good date after current_date
    for d in good_dates_list:
        if d > current_date:
            current_date = d
            break
    update_all_plots()


def change_period_length(delta: int):
    global current_period_length
    current_period_length_idx = list(period_lengths.values()).index(current_period_length)
    current_period_length_idx += delta
    if current_period_length_idx == len(period_lengths):
        current_period_length_idx = 0
    elif current_period_length_idx == -1:
        current_period_length_idx = len(period_lengths) - 1
    current_period_length = list(period_lengths.values())[current_period_length_idx]
    update_station_plot()


def change_a(delta: int):
    global current_b
    current_b_idx = list(b_values).index(current_b)
    current_b_idx += delta
    if current_b_idx == len(b_values):
        current_b_idx = 0
    elif current_b_idx == -1:
        current_b_idx = len(b_values) - 1
    current_b = b_values[current_b_idx]
    # a_text.set_text(f"a = {current_b}")
    update_all_plots()


def change_mb(delta: int):
    global current_member
    current_mb_idx = list(member_values).index(current_member)
    current_mb_idx += delta
    if current_mb_idx == len(member_values):
        current_mb_idx = 0
    elif current_mb_idx == -1:
        current_mb_idx = len(member_values) - 1
    current_member = member_values[current_mb_idx]

    update_all_plots()


# Event handler: update line plot when a point is clicked
def on_pick(event):
    global current_station_idx
    current_station_idx = event.ind[0]  # Index of the clicked point
    update_station_plot()


def update_spatial_plots():
    [ax.clear() for ax in axs_spatial]

    # Obs satellite
    s2_title = "Sentinel-2 FSC [-]"
    if np.datetime64(current_date) in snow_cover_s2.coords["time"]:
        s2_data = snow_cover_s2.sel(time=current_date)
        add_2d_plot(s2_data, ax_s2, dem_20m, title=s2_title, cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    else:
        ax_s2.set_title(s2_title)
        ax_s2.set_xticks([]), ax_s2.set_yticks([])
    if np.datetime64(current_date) in snow_cover_viirs.coords["time"]:
        viirs_data = snow_cover_viirs.sel(time=current_date)
        add_2d_plot(viirs_data, ax_viirs, dem_250m, "VIIRS FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    else:
        ax_viirs.set_title("VIIRS FSC [-]")
        ax_viirs.set_xticks([]), ax_viirs.set_yticks([])

    # Open loop
    sd_data = snow_depth_edel_ol_ds.sel(time=current_date, member=current_member)
    add_2d_plot(sd_data, ax_edel_ol_sd, dem_250m, "Edelweiss OL SD[m]", cmap=SNOW_DEPTH_CMAP)  # , vmin=0.001, vmax=2.5)

    fsc_edel = dickinson(sd=snow_depth_edel_ol_ds.sel(time=current_date, member=current_member), a=0.11, b=current_b)
    add_2d_plot(fsc_edel, ax_edel_ol_fsc, dem_250m, "Edelweiss OL FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    diff_edel_viirs = fsc_edel - snow_cover_viirs.sel(time=current_date)
    add_2d_plot(
        diff_edel_viirs,
        ax_diff_ol,
        dem_250m,
        f"Diff FSC Edelweiss OL - VIIRS [-] RMSE={compute_rmse(diff_edel_viirs):.2f}",
        cmap=FIELD_DIFF_CMAP,
        vmin=-1,
        vmax=1,
    )

    # Assim
    sd_data = snow_depth_edel_an_ds.sel(time=current_date, member=current_member)
    add_2d_plot(sd_data, ax_edel_an_sd, dem_250m, "Edelweiss assim SD [m]", cmap=SNOW_DEPTH_CMAP)  # , vmin=0.001, vmax=2.5)

    fsc_edel = dickinson(sd=snow_depth_edel_an_ds.sel(time=current_date, member=current_member), b=current_b, a=0.11)
    add_2d_plot(fsc_edel, ax_edel_an_fsc, dem_250m, "Edelweiss assim FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    diff_edel_viirs = fsc_edel - snow_cover_viirs.sel(time=current_date)
    add_2d_plot(
        diff_edel_viirs,
        ax_diff_an,
        dem_250m,
        f"Diff FSC Edelweiss assim - VIIRS [-] RMSE={compute_rmse(diff_edel_viirs):.2f}",
        cmap=FIELD_DIFF_CMAP,
        vmin=-1,
        vmax=1,
    )

    ######### FORCING ###########
    total_precip = forcing_ol.data_vars["precip_total"].sel(time=current_date, member=current_member)
    add_2d_plot(total_precip, ax_precip_ol, dem_250m, "Total precipitation OL [mm/day]", cmap=PRECIP_CMAP)

    phase_data = forcing_ol.data_vars["phase"].sel(time=current_date, member=current_member)
    add_2d_plot(phase_data, ax_phase_ol, dem_250m, "Precipitation phase OL [-]", cmap=PHASE_CMAP, vmin=-1, vmax=1)

    total_precip = forcing_an.data_vars["precip_total"].sel(time=current_date, member=current_member)
    add_2d_plot(total_precip, ax_precip_an, dem_250m, "Total precipitation assim [mm/day]", cmap=PRECIP_CMAP)

    phase_data = forcing_an.data_vars["phase"].sel(time=current_date, member=current_member)
    add_2d_plot(phase_data, ax_phase_an, dem_250m, "Precipitation phase assim [-]", cmap=PHASE_CMAP, vmin=-1, vmax=1)

    fig_maps.suptitle(str(current_date.date()), y=0.98)
    fig_maps.canvas.draw_idle()


def update_snowline_plots():
    [ax.clear() for ax in axs_snowlines]
    fig_snowlines.texts.clear()
    alt_max = snowline_ol_edel_ds.coords["altitude_max"].max()
    alt_min = snowline_ol_edel_ds.coords["altitude_min"].min()
    set_polarplot(ax=ax_snowlines, alt_max=alt_max, alt_min=alt_min)
    set_polarplot(ax=ax_snow_rain_line, alt_max=alt_max, alt_min=alt_min)
    # Determine ensemble visualization indexes
    plot_kwargs = {"ax": ax_snowlines}

    # Snowline
    current_edel_ol_sl = find_snowline_from_snow_penalization(snowline_ol_edel_ds.sel(time=current_date, a=current_b))
    plot_polar_envelop_member(current_edel_ol_sl, current_member, COLORS["edel_ol"], LABELS["edel_ol"], **plot_kwargs)

    current_edel_an_sl = find_snowline_from_snow_penalization(snowline_an_edel_ds.sel(time=current_date, a=current_b))
    plot_polar_envelop_member(current_edel_an_sl, current_member, COLORS["edel_an"], LABELS["edel_an"], **plot_kwargs)

    snowline_viirs = find_snowline_from_snow_penalization(snowline_viirs_ds.sel(time=current_date))
    plot_snowline_polarplot(snowline_viirs, label=LABELS["viirs"], color=COLORS["viirs"], **plot_kwargs)

    diff_snowline_viirs = current_edel_ol_sl.sel(member=current_member) - snowline_viirs
    fig_snowlines.text(x=0.1, y=0.95, s=f"RMSE OL - VIIRS: {compute_rmse(diff_snowline_viirs):.1f}")

    diff_snowline_viirs = current_edel_an_sl.sel(member=current_member) - snowline_viirs
    fig_snowlines.text(x=0.1, y=0.92, s=f"RMSE assim - VIIRS: {compute_rmse(diff_snowline_viirs):.1f}")

    if np.datetime64(current_date) in snowline_s2_ds.coords["time"]:
        snowline_s2 = find_snowline_from_snow_penalization(snowline_s2_ds.sel(time=current_date))
        plot_snowline_polarplot(snowline_s2, label=LABELS["s2"], color=COLORS["s2"], **plot_kwargs)

        diff_snowline_s2 = current_edel_ol_sl.sel(member=current_member) - snowline_s2
        fig_snowlines.text(x=0.1, y=0.89, s=f"RMSE OL - S2: {compute_rmse(diff_snowline_s2):.1f}")

        diff_snowline_s2 = current_edel_an_sl.sel(member=current_member) - snowline_s2
        fig_snowlines.text(x=0.1, y=0.86, s=f"RMSE assim - S2: {compute_rmse(diff_snowline_s2):.1f}")

    ax_snowlines.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))

    # Snow rain line
    plot_kwargs.update({"ax": ax_snow_rain_line})

    # When at least one of the aspects is not covered by the phase field (-> all aktitudes are NaNs) we do not plot
    if not np.any(np.isnan(snow_rain_forcing_ol_ds.sel(time=current_date).data_vars["phase"]).all(dim="altitude_bins")):
        current_forc_ol_srl = find_forcing_snowrain_line(snow_rain_forcing_ol_ds.sel(time=current_date))
        current_forc_an_srl = find_forcing_snowrain_line(snow_rain_forcing_an_ds.sel(time=current_date))
        plot_polar_envelop_member(current_forc_ol_srl, current_member, COLORS["edel_ol"], LABELS["edel_ol"], **plot_kwargs)
        plot_polar_envelop_member(current_forc_an_srl, current_member, COLORS["edel_an"], LABELS["edel_an"], **plot_kwargs)
    else:
        logger.info(f"no phase on day {current_date}")

    ax_snow_rain_line.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
    ax_snowlines.set_title("Snowline", va="bottom")
    ax_snow_rain_line.set_title("Snow rain line", va="bottom")
    fig_snowlines.suptitle(str(current_date.date()))
    fig_snowlines.canvas.draw_idle()


def update_station_plot():
    [ax.clear() for ax in axs_stations]
    fig_stations.texts.clear()
    x_station, y_station = x_poste[current_station_idx], y_poste[current_station_idx]

    # Maps
    sd_edel_data = snow_depth_edel_an_ds.sel(time=current_date, member=current_member)
    add_2d_plot(sd_edel_data, ax_station_map_sd, dem_250m, "Edelweiss assim SD [m]", cmap=SNOW_DEPTH_CMAP)
    fsc_edel_data = dickinson(sd_edel_data, b=current_b, a=0.11)
    add_2d_plot(fsc_edel_data, ax_station_map_fsc, dem_250m, "Edelweiss assim FSC [-]", cmap=FSC_CMAP_SNOW_COVER)
    ax_station_map_sd.plot(imshow_locations[:, 0], imshow_locations[:, 1], linewidth=0, marker="*", color="y", picker=5)

    period = slice(current_date, current_date + current_period_length)
    # BDclim
    sd_time_station = (
        bdclim.sel(time=period).where(bdclim["x"] == x_station, drop=True).where(bdclim["y"] == y_station, drop=True)
    )
    station_name = sd_time_station.coords["Station_Name"].values[0]
    altitude = sd_time_station.coords["ZS"].values[0]
    t_coord_bdclim = sd_time_station.coords["time"]

    ax_station_time_sd.plot(
        t_coord_bdclim, sd_time_station.values, color="black", linestyle="dashed", linewidth=2, label="in situ"
    )
    fsc_time_poste = dickinson(sd_time_station, b=current_b, a=0.11)
    ax_station_time_fsc.plot(
        t_coord_bdclim, fsc_time_poste.values, color="black", linestyle="dashed", linewidth=2, label="in situ"
    )

    ax_station_time_sd.set_title(f"{station_name} - alt. {altitude}m - Snow depth [m]")
    ax_station_time_fsc.set_title(f"{station_name} - alt. {altitude}m - FSC [-]")

    # SD Open loop
    sd_edel_station_ol = snow_depth_edel_ol_ds.sel(time=period).sel(x=x_station, y=y_station, method="nearest")
    plot_ensemble_time_series(
        sd_edel_station_ol, mb_to_plot=current_member, ax=ax_station_time_sd, color=COLORS["edel_ol"], label=LABELS["edel_ol"]
    )
    diff_edel_insitu = sd_edel_station_ol.sel(member=current_member) - sd_time_station
    fig_stations.text(x=0.15, y=0.37, s=f"RMSE OL - in situ: {compute_rmse(diff_edel_insitu):.2f}")

    # SD Assim
    sd_edel_station_an = snow_depth_edel_an_ds.sel(time=period).sel(x=x_station, y=y_station, method="nearest")
    plot_ensemble_time_series(
        sd_edel_station_an, mb_to_plot=current_member, ax=ax_station_time_sd, color=COLORS["edel_an"], label=LABELS["edel_an"]
    )
    diff_edel_insitu = sd_edel_station_an.sel(member=current_member) - sd_time_station
    fig_stations.text(x=0.3, y=0.37, s=f"RMSE assim - in situ: {compute_rmse(diff_edel_insitu):.2f}")

    # FSC Open loop
    fsc_edel_station_ol = dickinson(sd_edel_station_ol, b=current_b, a=0.11)
    plot_ensemble_time_series(
        fsc_edel_station_ol,
        mb_to_plot=current_member,
        ax=ax_station_time_fsc,
        color=COLORS["edel_ol"],
        label=LABELS["edel_ol"],
    )
    diff_edel_insitu = fsc_edel_station_ol.sel(member=current_member) - fsc_time_poste
    fig_stations.text(x=0.15, y=0.06, s=f"RMSE OL - in situ: {compute_rmse(diff_edel_insitu):.2f}")

    # FSC Assim
    fsc_edel_station_an = dickinson(sd_edel_station_an, b=current_b, a=0.11)
    plot_ensemble_time_series(
        fsc_edel_station_an,
        mb_to_plot=current_member,
        ax=ax_station_time_fsc,
        color=COLORS["edel_an"],
        label=LABELS["edel_an"],
    )
    diff_edel_insitu = fsc_edel_station_an.sel(member=current_member) - fsc_time_poste
    fig_stations.text(x=0.3, y=0.06, s=f"RMSE assim - in situ: {compute_rmse(diff_edel_insitu):.2f}")

    ax_station_time_sd.grid(True)
    ax_station_time_fsc.set_ylim(-0.1, 1.1)
    ax_station_time_fsc.grid(True)

    # Pleaides
    sd_pleiades_station = pleiades.sel(time=period).sel(x=x_station, y=y_station, method="nearest")
    ax_station_time_sd.plot(
        sd_pleiades_station.time,
        sd_pleiades_station.values,
        marker="*",
        linewidth=0,
        markersize=15,
        color="gold",
        label="Pleiades",
    )

    # Sentinel-2
    trans = Transformer.from_crs(2154, 32631)
    x_station_s2, y_station_s2 = trans.transform(x_station, y_station)
    snow_cover_s2_station = (
        snow_cover_s2.sel(time=good_dates_s2).sel(time=period).sel(x=x_station_s2, y=y_station_s2, method="nearest")
    )
    ax_station_time_fsc.plot(
        snow_cover_s2_station.time + np.timedelta64(10, "h") + np.timedelta64(30, "m"),  # Sentinel-2 10:30 am pass
        snow_cover_s2_station,
        marker="s",
        mfc="none",
        linewidth=0,
        markersize=10,
        color=COLORS["s2"],
        label=LABELS["s2"],
    )

    # VIIRS
    snow_cover_viirs_station = (
        snow_cover_viirs.sel(time=good_dates_viirs).sel(time=period).sel(x=x_station, y=y_station, method="nearest")
    )
    ax_station_time_fsc.plot(
        snow_cover_viirs_station.time + np.timedelta64(12, "h"),  # Observation assimilated at 12h
        snow_cover_viirs_station,
        marker="o",
        linewidth=0,
        markersize=8,
        color=COLORS["viirs"],
        label=LABELS["viirs"],
    )

    # Plot a bar corresponding to the date on the time series
    ax_station_time_sd.plot(
        [current_date, current_date], ax_station_time_sd.get_ybound(), linewidth=3, color="orange", linestyle=(0, (1, 1))
    )
    ax_station_time_fsc.plot(
        [current_date, current_date], ax_station_time_fsc.get_ybound(), linewidth=3, color="orange", linestyle=(0, (1, 1))
    )

    ax_station_time_sd.legend()
    ax_station_time_fsc.legend()
    current_period_length_idx = list(period_lengths.values()).index(current_period_length)
    fig_stations.text(s=f"Period = {list(period_lengths.keys())[current_period_length_idx]}", y=0.38, x=0.03)
    fig_stations.suptitle(str(current_date.date()))
    fig_stations.canvas.draw()


def update_all_plots():

    update_spatial_plots()
    update_snowline_plots()
    update_station_plot()
    fig_buttons.texts.clear()
    fig_buttons.text(s=f"a = {current_b}", y=0.65, x=0.7)
    fig_buttons.text(s=f"member = {'avg.' if current_member == -1 else current_member}", y=0.85, x=0.7)

    fig_buttons.canvas.draw_idle()
    fig_stations.canvas.mpl_connect("pick_event", on_pick)


# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    ################################ User inputs #############################################
    xpid = "assim_viirs_all_clear_dates_november_2021"
    working_folder = "/home/imperatoren/work/edelweiss_assimilation/"
    observation_folder = f"{working_folder}/observations/grandesrousses250m"
    simulation_folder = f"{working_folder}/simulations/postprocess"
    s2_folder = f"{observation_folder}/s2"
    edelweiss_ol_folder = f"{simulation_folder}/grandesrousses250m/open_loop/"
    edelweiss_an_folder = f"{simulation_folder}/reanalysis/{xpid}/"
    viirs_folder = f"{observation_folder}/meteofrance/"
    forcing_ol_folder = f"{working_folder}/forcing/grandesrousses250m/open_loop"
    forcing_analysis_folder = f"{working_folder}/forcing/reanalysis/{xpid}/"
    bdclim_filepath = f"{working_folder}/observations/grandesrousses250m/bdclim/bdclim.nc"
    pleiades_path = f"{observation_folder}/pleiades_grandesrousses_all.nc"

    topography_data_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/topography/"
    landcover_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/"
    forest_mask_path = f"{landcover_folder}/forest_mask/forest_mask_corine_grandesrousses_max.nc"
    glacier_mask_path = f"{landcover_folder}/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"
    dem_250m_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    dem_20m_filepath = f"{topography_data_folder}/20m/DEM_GR_UTM_20m.tif"
    # Snowline plots
    LABELS = {"s2": "Sentinel-2", "viirs": "VIIRS", "edel_ol": "Edelweiss OL", "edel_an": "Edelweiss assim"}
    COLORS = {"s2": "black", "viirs": "red", "edel_ol": "blue", "edel_an": "magenta"}

    # Initial date
    current_date = datetime(2021, 11, 1)
    period_length = timedelta(days=30)
    date_end = current_date + period_length

    fig_buttons = plt.figure(figsize=(8, 2))
    buttons = InteractiveSeasonExploreButtons(fig=fig_buttons)
    buttons.btn_d_minus.on_clicked(lambda e: change_day(-1))
    buttons.btn_d_plus.on_clicked(lambda e: change_day(1))

    buttons.btn_m_minus.on_clicked(lambda e: change_month(-1))
    buttons.btn_m_plus.on_clicked(lambda e: change_month(1))

    buttons.btn_next_good_viirs.on_clicked(lambda e: next_good_date(good_dates_viirs))
    buttons.btn_prev_good_viirs.on_clicked(lambda e: last_good_date(good_dates_viirs))
    buttons.btn_next_good_s2.on_clicked(lambda e: next_good_date(good_dates_s2))
    buttons.btn_prev_good_s2.on_clicked(lambda e: last_good_date(good_dates_s2))

    buttons.btn_a_minus.on_clicked(lambda e: change_a(-1))
    buttons.btn_a_plus.on_clicked(lambda e: change_a(1))

    buttons.btn_mb_minus.on_clicked(lambda e: change_mb(-1))
    buttons.btn_mb_plus.on_clicked(lambda e: change_mb(1))

    ########################### Read data ##########################################################

    logger.info("Reading data")
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

    snow_depth_edel_ol_ds = xr.open_dataset(f"{edelweiss_ol_folder}/spatial.nc")
    snow_depth_edel_ol_ds = snow_depth_edel_ol_ds.data_vars["DSN_T_ISBA"].sortby("y", ascending=False).where(1 - mask)
    snow_depth_edel_an_ds = xr.open_dataset(f"{edelweiss_an_folder}/spatial.nc")
    snow_depth_edel_an_ds = snow_depth_edel_an_ds.data_vars["DSN_T_ISBA"].sortby("y", ascending=False).where(1 - mask)

    snowline_s2_ds = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_ol_edel_ds = xr.open_dataset(f"{edelweiss_ol_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_an_edel_ds = xr.open_dataset(f"{edelweiss_an_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_viirs_ds = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")

    forcing_ol = xr.open_mfdataset(f"{forcing_ol_folder}/spatial.nc").sortby("y", ascending=False).where(1 - mask)
    forcing_an = xr.open_mfdataset(f"{forcing_analysis_folder}/spatial.nc").sortby("y", ascending=False).where(1 - mask)

    snow_rain_forcing_ol_ds = xr.open_dataset(f"{forcing_ol_folder}/snowline_parametrization.nc").sel(slope_bins="8 - 30")
    snow_rain_forcing_an_ds = xr.open_dataset(f"{forcing_analysis_folder}/snowline_parametrization.nc").sel(
        slope_bins="8 - 30"
    )

    # Independent observations
    start_time, end_time = snow_depth_edel_ol_ds.coords["time"][0], snow_depth_edel_ol_ds.coords["time"][-1]
    bdclim = xr.open_dataarray(bdclim_filepath).sel(time=slice(start_time, end_time))
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim)
    pleiades = (xr.open_dataset(pleiades_path).where(1 - mask).sel(time=slice(start_time, end_time))).data_vars["snow_depth"]

    good_dates_s2 = find_clear_dates_s2(snow_cover_s2=snow_cover_s2)
    good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=snow_cover_viirs)

    # values for observation operator values
    b_values = snowline_ol_edel_ds.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_b = b_values[0]

    # Values for member values
    member_values = snowline_ol_edel_ds.coords["member"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_member = member_values[0]

    # Values for member values
    period_lengths = {
        "1 day": timedelta(days=1),
        "1 week": timedelta(weeks=1),
        "1 month": timedelta(days=30),
        "2 months": timedelta(days=60),
        "6 months": timedelta(days=180),
        "1 year": timedelta(days=365),
    }
    # Initial value
    current_period_length = period_lengths["1 month"]
    # Initial value for observation operator parametrization
    current_member = member_values[0]

    logger.info("Plotting")
    fig_maps = plt.figure(figsize=(20, 10))
    ax_edel_ol_sd = fig_maps.add_subplot(3, 4, 1)
    ax_edel_ol_fsc = fig_maps.add_subplot(3, 4, 2)
    ax_precip_ol = fig_maps.add_subplot(3, 4, 3)
    ax_phase_ol = fig_maps.add_subplot(3, 4, 4)

    ax_edel_an_sd = fig_maps.add_subplot(3, 4, 5)
    ax_edel_an_fsc = fig_maps.add_subplot(3, 4, 6)
    ax_precip_an = fig_maps.add_subplot(3, 4, 7)
    ax_phase_an = fig_maps.add_subplot(3, 4, 8)

    ax_s2 = fig_maps.add_subplot(3, 4, 9)
    ax_viirs = fig_maps.add_subplot(3, 4, 10)
    ax_diff_ol = fig_maps.add_subplot(3, 4, 11)
    ax_diff_an = fig_maps.add_subplot(3, 4, 12)

    axs_ol = [ax_edel_ol_sd, ax_edel_ol_fsc, ax_precip_ol, ax_phase_ol]
    axs_an = [ax_edel_an_sd, ax_edel_an_fsc, ax_precip_an, ax_phase_an]
    axs_spatial = [*axs_ol, *axs_an, ax_viirs, ax_s2, ax_diff_ol, ax_diff_an]

    fig_snowlines = plt.figure(figsize=(6, 10))
    # Put the figure next to station comparison
    fig_snowlines.canvas.manager.window.wm_geometry("+1500+0")

    ax_snowlines = fig_snowlines.add_subplot(2, 1, 1, projection="polar")
    ax_snow_rain_line = fig_snowlines.add_subplot(2, 1, 2, projection="polar")
    axs_snowlines = [ax_snowlines, ax_snow_rain_line]

    ## Station plot
    fig_stations = plt.figure(figsize=(132, 12))
    # Put the figure on the left upper cotner of the screen
    fig_stations.canvas.manager.window.wm_geometry("+0+0")
    grid_spec = GridSpec(3, 2, figure=fig_stations)
    ax_station_map_sd = fig_stations.add_subplot(grid_spec[0, 0])
    ax_station_map_fsc = fig_stations.add_subplot(grid_spec[0, 1])
    ax_station_time_sd = fig_stations.add_subplot(grid_spec[1, :])
    ax_station_time_fsc = fig_stations.add_subplot(grid_spec[2, :])

    current_station_idx = 0
    axs_stations = [ax_station_map_sd, ax_station_map_fsc, ax_station_time_sd, ax_station_time_fsc]
    x_coords_stations, y_coords_stations = find_station_locations(bdclim_dataset=bdclim)

    imshow_locations = np.array(
        [
            find_station_imshow_location(x, y, snow_depth_edel_an_ds["x"], snow_depth_edel_an_ds["y"])
            for x, y in zip(x_coords_stations, y_coords_stations)
        ]
    )
    ax_change_period_length = fig_stations.add_axes([0.03, 0.33, 0.07, 0.03])
    button_change_period_length = Button(ax_change_period_length, "Change period")
    button_change_period_length.on_clicked(lambda e: change_period_length(1))

    # Date title
    fig_maps.suptitle(str(current_date.date()), y=0.98)
    fig_snowlines.suptitle(str(current_date.date()), y=0.98)
    fig_stations.suptitle(str(current_date.date()), y=0.98)

    fig_stations.canvas.mpl_connect("pick_event", on_pick)
    # Initial plot
    fig_maps.subplots_adjust(bottom=0.05, top=0.96, left=0.05, hspace=0.04, right=0.96, wspace=0.05)
    fig_stations.subplots_adjust(hspace=0.3, top=0.96, wspace=0.0)

    update_all_plots()
    # plt.tight_layout()

    plt.show()
