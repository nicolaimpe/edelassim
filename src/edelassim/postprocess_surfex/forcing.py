import numpy as np
import xarray as xr


def reorder_forcing_by_duplicated_particles(
    forcing: xr.Dataset, duplicated_particles: xr.DataArray, start_assim_date: str | None = None
) -> xr.Dataset:
    """_summary_

    Args:
        forcing (xr.Dataset): see scripts/process_forcing.py
        duplicated_particles (xr.DataArray): see postprocess_surfex.soda.duplicated_particles_from_part_file
        start_assim_date (str | None, optional): if different from forcing initial date. Defaults to None.

    Returns:
        xr.Dataset: equivalent analysis forcing
    """
    # Transpose to match with duplicated particles
    forcing = forcing.stack(n_points=("y", "x")).transpose("time", "n_points", "member")
    forcing_reordered = forcing.copy(deep=True)
    old_assim_date = start_assim_date if start_assim_date is not None else forcing.time[0]
    for assim_date in duplicated_particles.time:
        forcing_assim_window = forcing.sel(time=slice(old_assim_date, assim_date))
        member_idxs_time_window = duplicated_particles.sel(time=assim_date).expand_dims("time") - 1
        member_idxs_time_window_repeated = xr.DataArray(
            data=np.repeat(member_idxs_time_window.values, repeats=forcing_assim_window.sizes["time"], axis=0),
            dims=duplicated_particles.dims,
            coords={
                "time": forcing_assim_window.coords["time"],
                "n_points": forcing_assim_window.coords["n_points"],
                "member": forcing_assim_window.coords["member"],
            },
        )
        for dv in forcing_reordered.data_vars:
            forcing_reordered.sel(time=slice(old_assim_date, assim_date)).data_vars[dv][:] = (
                forcing_assim_window.data_vars[dv].sel(member=member_idxs_time_window_repeated).values
            )
        old_assim_date = assim_date

    forcing_reordered = forcing_reordered.unstack()
    return forcing_reordered
