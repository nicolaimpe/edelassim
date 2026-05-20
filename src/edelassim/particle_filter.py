import numpy as np
import xarray as xr


def point_particle_filter(y_hat: xr.DataArray, y: xr.DataArray, inv_R: xr.DataArray) -> xr.DataArray:
    innovation = y - y_hat
    likelihood = np.exp(-1 / 2 * (innovation * inv_R * innovation))
    evidence = likelihood.sum(dim="member")
    new_weigths = likelihood / evidence
    return new_weigths


def effective_weigths(weights: xr.DataArray):
    return 1 / (weights**2).sum(dim="member")


def klocal_particle_filter(k_correlations: xr.DataArray, y_hat: xr.DataArray, y: xr.DataArray, inv_R: xr.DataArray):
    n_eff_weights = 0
    k = k_correlations.sizes["j"]
    while n_eff_weights < 8 and k > 1:
        k = k - 1
        k_idxs = k_correlations.grid_point_j.values[0][:k]

        local_y_hat = y_hat.sel(grid_point_j=k_idxs)
        local_y = y.sel(grid_point_j=k_idxs)
        local_inv_R = inv_R.sel(grid_point_j=k_idxs)

        innovation = local_y - local_y_hat
        likelihood = np.exp(-1 / 2 * (innovation * local_inv_R * innovation)).sum(dim="grid_point_j")
        evidence = likelihood.sum(dim="member")
        out_weigths = likelihood / evidence

        n_eff_weights = effective_weigths(out_weigths)

    return out_weigths


avg_sigma_obs = 0.15
tets_inv_R = 1 / avg_sigma_obs**2 * np.ones(shape=(stacked_data.sizes["n_point"],))

test_obs_stacked = observation.data_vars["SCF"].stack(n_point=("x", "y"))
test_inv_R = xr.DataArray(tets_inv_R, coords=test_obs_stacked.coords)
test_y_hat = zaitchik(swe=stacked_data)

grid_points = np.arange(1, len(stacked_data["n_point"]) + 1)
test_obs_stacked = test_obs_stacked.assign_coords({"grid_point_j": ("n_point", grid_points)}).swap_dims(
    {"n_point": "grid_point_j"}
)
test_y_hat = test_y_hat.assign_coords({"grid_point_j": ("n_point", grid_points)}).swap_dims({"n_point": "grid_point_j"})
test_inv_R = test_inv_R.assign_coords({"grid_point_j": ("n_point", grid_points)}).swap_dims({"n_point": "grid_point_j"})
weights_klocal = sorted_correlations.groupby("grid_point_i").map(
    klocal_particle_filter, (test_y_hat, test_obs_stacked, test_inv_R)
)


new_weigths_point_pf = point_particle_filter(y_hat=test_y_hat, y=test_obs_stacked, inv_R=test_inv_R)
n_eff = effective_weigths(weights=new_weigths_point_pf)
n_eff.unstack().plot.imshow(x="x")


# Retrieve spatial representation by "manually" setting coordinates...this should be better handle
weights_klocal_spatial = (
    weights_klocal.assign_coords(
        {
            "n_point": ("grid_point_i", test_y_hat.coords["n_point"].values),
            "x": ("grid_point_i", test_y_hat.coords["x"].values),
            "y": ("grid_point_i", test_y_hat.coords["y"].values),
        }
    )
    .swap_dims({"grid_point_i": "n_point"})
    .set_index({"n_point": ("x", "y")})
    .unstack()
)
n_eff_klocal = effective_weigths(weights=weights_klocal_spatial)
n_eff_klocal.plot.imshow(x="x")
