from glob import glob

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.gridspec import GridSpec

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset
from edelassim.filter import effective_sample_size_xarray
from edelassim.grids import EdelweissGrandesRoussesGrid
from edelassim.postprocess_surfex.soda import alpha_multiple_assimilation, alpha_to_grid, duplicated_particles_to_grid
from edelassim.visualization.static_files import (
    get_analysis_station_scores,
    get_bdclim,
    get_duplicated_particles_assim_dates,
    get_open_loop_station_scores,
    working_folder,
)

if __name__ == "__main__":
    ### Inputs and preprocessing
    xpids = ["assim_viirs_cloudcover055_wy2122_python_pf", "assim_viirs_cloudcover055_wy2122_sigma_obs_var"]
    plot_colors = ["magenta", "darkgreen"]
    plot_labels = ["sigma const", "sigma var"]
    period = slice("2021-11", "2022-06")
    bdclim = get_bdclim(period=period)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Edelweiss vs in situ ({bdclim.sizes['num_poste']} locations aggregated) ")
    score_name = "crps"
    grid_spec = GridSpec(nrows=4, ncols=1, hspace=0.25, height_ratios=(1, 1, 1, 1))
    ax_scores_sd = fig.add_subplot(grid_spec[0])
    ax_scores_sd.set_title(f"SD {score_name.upper()}S")
    ax_scores_fsc = fig.add_subplot(grid_spec[1])
    ax_scores_fsc.set_title(f"FSC {score_name.upper()}S")

    sd_ol_scores, fsc_ol_scores = get_open_loop_station_scores(period=period)
    # print(sd_ol_scores.data_vars[score_name])
    mean_sd_ol_score = sd_ol_scores.data_vars[score_name].sel(num_poste=bdclim.coords["num_poste"]).mean(dim="num_poste")
    mean_fsc_ol_score = fsc_ol_scores.data_vars[score_name].sel(num_poste=bdclim.coords["num_poste"]).mean(dim="num_poste")
    for idx, xpid in enumerate(xpids):
        sd_assim_scores, fsc_assim_scores = get_analysis_station_scores(xpid=xpid, period=period)
        mean_sd_assim_score = (
            sd_assim_scores.data_vars[score_name].sel(num_poste=bdclim.coords["num_poste"]).mean(dim="num_poste")
        )
        crpss_sd = 1 - mean_sd_assim_score / mean_sd_ol_score
        ax_scores_sd.plot(
            crpss_sd.coords["time"],
            crpss_sd.values,
            linewidth=0,
            marker="^",
            color=plot_colors[idx],
            fillstyle="none",
            label=f"{plot_labels[idx]}",
        )

        mean_fsc_assim_score = (
            fsc_assim_scores.data_vars[score_name].sel(num_poste=bdclim.coords["num_poste"]).mean(dim="num_poste")
        )
        crpss_fsc = 1 - mean_fsc_assim_score / mean_fsc_ol_score
        ax_scores_fsc.plot(
            crpss_fsc.coords["time"],
            crpss_fsc.values,
            linewidth=0,
            marker="o",
            color=plot_colors[idx],
            fillstyle="none",
            label=f"{plot_labels[idx]}",
        )

    ## Finish plot
    [ax.set_ylim(-1, 1) for ax in fig.get_axes()]

    ## Ensemble monitoring
    ax_ess = fig.add_subplot(grid_spec[2])
    ax_alpha = fig.add_subplot(grid_spec[3])
    for idx, xpid in enumerate(xpids):
        duplicated_particles, _ = get_duplicated_particles_assim_dates(acutal_xpid=xpid)
        duplicated_particles = duplicated_particles_to_grid(duplicated_particles, EdelweissGrandesRoussesGrid())
        duplicated_particles_station = extract_bdclim_locations_on_spatial_dataset(
            bdclim, spatial_dataset=duplicated_particles
        )
        particle_occurrence = xr.zeros_like(duplicated_particles_station)

        # Compute approximate sample size from duplicated particles
        for mb in duplicated_particles.coords["member"]:
            n = (duplicated_particles_station == mb + 1).sum(dim=("member"))
            particle_occurrence[{"member": mb}] = n
        # Normalized Effective sample size mean
        n_members = particle_occurrence.sizes["member"]
        eff_sample_size = effective_sample_size_xarray(weights=(particle_occurrence / n_members))
        global_ess = eff_sample_size.mean(dim=("num_poste"))
        ax_ess.plot(
            global_ess.time,
            global_ess.values / n_members,
            linewidth=0,
            marker="*",
            color=plot_colors[idx],
            label=f"ESS/Ne {plot_labels[idx]}",
        )

        # Alpha mean

        alpha_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*ALPHA*")
        alpha_coefficients = alpha_to_grid(
            alpha_multiple_assimilation(alpha_files=alpha_files), grid=EdelweissGrandesRoussesGrid()
        )
        alpha_station = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=alpha_coefficients)
        alpha_mean = alpha_station.mean(dim=("num_poste"))
        ax_alpha.plot(
            alpha_mean.time,
            alpha_mean,
            linewidth=0,
            marker="s",
            fillstyle="none",
            color=plot_colors[idx],
            label=f"alpha {plot_labels[idx]}",
        )

    ax_alpha.set_xticks(alpha_mean.time, minor=True)
    ax_alpha.grid(which="minor")
    ax_alpha.tick_params(axis="x", which="minor", bottom=False)
    ax_alpha.set_title("Alpha mean on station points")

    ax_ess.set_xticks(global_ess.time, minor=True)
    ax_ess.grid(which="minor")
    ax_ess.tick_params(axis="x", which="minor", bottom=False)
    ax_ess.set_title("ESS/N_e on station points")

    ## Finish scores
    [ax.legend() for ax in fig.get_axes()]
    [ax.grid() for ax in fig.get_axes()]
    plt.show()
