import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from edelassim.observation_operators import dickinson
from edelassim.visualization.spatial import SNOW_DEPTH_CMAP, add_2d_plot
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-06")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, pleiades, bdclim = get_spatial_datasets(xpid)
    pleiades = pleiades.sel(time=period).sel(band=1).sortby("y", ascending=False)
    dem = xr.open_dataarray(f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif").sel(band=1)
    edelweiss_open_loop_pleiades_dates = (
        edelweiss_openloop.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"]).sortby("y", ascending=False)
    )
    edelweiss_analysis_pleiades_dates = (
        edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"]).sortby("y", ascending=False)
    )
    grouped_dataset = xr.Dataset(
        {
            "open_loop": edelweiss_open_loop_pleiades_dates.sel(member=slice(0, None)),
            "analysis": edelweiss_analysis_pleiades_dates.sel(member=slice(0, None)),
            "pleiades": pleiades.data_vars["snow_depth"],
        },
        coords=edelweiss_open_loop_pleiades_dates.coords,
    )

    # def compute_crpss_pleiades(data: xr.Dataset) -> xr.DataArray:

    #     crps_open_loop = EnsembleScores(
    #         timeObs=None, timeSim=None, obs=data.data_vars["pleiades"].values, ensemble=data.data_vars["open_loop"].values
    #     ).CRPS()
    #     # print(data.data_vars["open_loop"].values)
    #     crps_analysis = EnsembleScores(
    #         timeObs=None, timeSim=None, obs=data.data_vars["pleiades"].values, ensemble=data.data_vars["analysis"].values
    #     ).CRPS()
    #     crpss = 1 - (crps_analysis / crps_open_loop)
    #     return xr.Dataset({"crps_open_loop": crps_open_loop, "crps_analysis": crps_analysis, "crpss": crpss})

    # crpss = grouped_dataset.groupby(("y", "x", "time")).map(compute_crpss_pleiades)
    # crpss.to_netcdf("crpss.nc")

    crpss = xr.open_dataset("../output_folder/crpss.nc").sortby("y", ascending=False)

    for i in (0, 1):
        fig, axs = plt.subplots(2, 3, figsize=(18, 18))
        fig.subplots_adjust(wspace=0.1, hspace=0.2)
        date_str = pd.Timestamp(pleiades.coords["time"].values[i]).strftime("%Y-%m-%d")

        # Snow depth maps and differences
        fig.suptitle(f"Edelweiss snow depth vs Pleiades - {date_str}")
        edel_ol = edelweiss_open_loop_pleiades_dates.isel(time=i).sel(member=-2)
        edel_an = edelweiss_analysis_pleiades_dates.isel(time=i).sel(member=-2)
        pleiades_obs = pleiades.data_vars["snow_depth"].isel(time=i)

        add_2d_plot(
            data=edel_ol, ax=axs[0, 0], dem=dem, title="Edelweiss median open loop", cmap=SNOW_DEPTH_CMAP, vmax=3, vmin=0.01
        )
        add_2d_plot(
            data=edel_an, ax=axs[0, 1], dem=dem, title="Edelweiss median assim", cmap=SNOW_DEPTH_CMAP, vmax=3, vmin=0.01
        )
        add_2d_plot(data=pleiades_obs, ax=axs[0, 2], dem=dem, title="Pleiades", cmap=SNOW_DEPTH_CMAP, vmax=3, vmin=0.01)
        title = "Edelweiss open loop median - pleiades difference"
        add_2d_plot(data=edel_ol - pleiades_obs, ax=axs[1, 0], dem=dem, title=title, cmap="coolwarm_r", vmax=1.5, vmin=-1.5)
        title = "Edelweiss assim median - pleiades difference"
        add_2d_plot(data=edel_an - pleiades_obs, ax=axs[1, 1], dem=dem, title=title, cmap="coolwarm_r", vmax=1.5, vmin=-1.5)
        title = "Edelweiss assim - open loop median"
        add_2d_plot(data=edel_an - edel_ol, ax=axs[1, 2], dem=dem, title=title, cmap="coolwarm_r", vmax=1.5, vmin=-1.5)

        # Ensemble scores and differences
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        fig.subplots_adjust(wspace=0.02, hspace=0.2)
        fig.suptitle(f"Edelweiss snow depth vs Pleiades ensemble scores - {date_str}")
        crps_ol = crpss.data_vars["crps_open_loop"].isel(time=i)
        crps_an = crpss.data_vars["crps_analysis"].isel(time=i)
        crps_score = crpss.data_vars["crpss"].isel(time=i)

        edel_ol_ensemble = edelweiss_open_loop_pleiades_dates.isel(time=i).sel(member=slice(0, None))
        edel_an_ensemble = edelweiss_analysis_pleiades_dates.isel(time=i).sel(member=slice(0, None))

        mae_open_loop = np.abs(edel_ol_ensemble - pleiades_obs).sel(member=slice(0, None)).mean(dim="member")
        mae_analysis = np.abs(edel_an_ensemble - pleiades_obs).sel(member=slice(0, None)).mean(dim="member")

        residuals_ensemble_open_loop = edel_ol_ensemble - edel_an_ensemble.mean(dim="member")
        residuals_ensemble_analysis = edel_an_ensemble - edel_an_ensemble.mean(dim="member")

        sigma_ensemble_open_loop = np.sqrt(((residuals_ensemble_open_loop) ** 2).mean(dim="member"))
        sigma_ensemble_analysis = np.sqrt(((residuals_ensemble_analysis) ** 2).mean(dim="member"))
        title = "Edelweiss open loop CRPS"
        add_2d_plot(data=crps_ol, ax=axs[0, 0], dem=dem, title=title, cmap="viridis", vmin=0, vmax=3)
        title = "Edelweiss assim CRPS"
        add_2d_plot(data=crps_an, ax=axs[0, 1], dem=dem, title=title, cmap="viridis", vmin=0, vmax=3)
        title = "CRPSS"
        add_2d_plot(data=crps_score, ax=axs[0, 2], dem=dem, title=title, cmap="coolwarm_r", vmax=1, vmin=-1)
        title = "Edelweiss open loop MAE"
        add_2d_plot(data=mae_open_loop, ax=axs[1, 0], dem=dem, title=title, cmap="viridis", vmin=0, vmax=2)
        title = "Edelweiss assim MAE"
        add_2d_plot(data=mae_analysis, ax=axs[1, 1], dem=dem, title=title, cmap="viridis", vmin=0, vmax=2)
        title = "Edelweiss open loop - assim MAE"
        add_2d_plot(
            data=mae_open_loop - mae_analysis, ax=axs[1, 2], dem=dem, title=title, cmap="coolwarm_r", vmin=-0.5, vmax=0.5
        )
        title = "Edelweiss open loop sigma"
        add_2d_plot(data=sigma_ensemble_open_loop, ax=axs[2, 0], dem=dem, title=title, cmap="viridis", vmin=0, vmax=1)
        title = "Edelweiss assim sigma"
        add_2d_plot(data=sigma_ensemble_analysis, ax=axs[2, 1], dem=dem, title=title, cmap="viridis", vmin=0, vmax=1)
        title = "Edelweiss open loop - assim sigma"
        add_2d_plot(
            data=sigma_ensemble_open_loop - sigma_ensemble_analysis,
            ax=axs[2, 2],
            dem=dem,
            title=title,
            cmap="coolwarm_r",
            vmin=-0.5,
            vmax=0.5,
        )

        # Zoom on CRPSS
        fig, ax = plt.subplots(figsize=(14, 10))
        add_2d_plot(data=crps_score, ax=ax, dem=dem, title=title, cmap="coolwarm_r", vmax=1, vmin=-1)
        ax.set_title(f"CRPSS Edelweiss snow depth vs Pleiades - {date_str}")

    ## VIIRS
    import matplotlib

    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Greens"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))

    newcmp = matplotlib.colors.ListedColormap(newcolors, name="RdGn")
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)

    fig, ax = plt.subplots(figsize=(14, 10))

    mae_open_loop = (
        np.abs(
            dickinson(edelweiss_openloop.data_vars["DSN_T_ISBA"]).sel(member=slice(0, None))
            - viirs.data_vars["snow_cover_fraction"]
        )
        .mean(dim="member")
        .mean(dim="time")
    )
    mae_analysis = (
        np.abs(
            dickinson(edelweiss_analysis.data_vars["DSN_T_ISBA"]).sel(member=slice(0, None))
            - viirs.data_vars["snow_cover_fraction"]
        )
        .mean(dim="member")
        .mean(dim="time")
    )
    newcmp = matplotlib.colors.ListedColormap(newcolors, name="RdGn")

    title = "Open loop - assim MAE - FSC Edelweiss vs VIIRS D+1 (not assimilated)"
    add_2d_plot(data=mae_open_loop - mae_analysis, ax=ax, dem=dem, title=title, cmap=newcmp, vmin=-0.2, vmax=0.2)
    plt.show()
