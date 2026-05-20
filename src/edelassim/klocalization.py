import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from edelassim.observation_operators import zaitchik
from edelassim.postprocess_surfex.prep import compute_all_members_snow_tickness_and_mass


def klocalize(correlations_point_i: xr.DataArray, k_max: int):
    sorted = correlations_point_i[0].sortby(lambda x: x, ascending=False)
    out = sorted.sel(j=slice(1, k_max + 1))
    return out


slope_map = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/slope.tif"
prep_files = []
folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/assim_viirs_local"
for i in range(1, 18):
    prep_files.append(f"{folder}/PREP_220226H12_PF_ENS{i}.nc")

# xr.open_dataset("/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/assim_viirs_local/")
all_member_sd_swe = compute_all_members_snow_tickness_and_mass(prep_files=prep_files, slope_file=slope_map)
scf = xr.DataArray(
    data=zaitchik(swe=all_member_sd_swe.data_vars["swe"].values),
    name="scf",
    coords=all_member_sd_swe.coords,
    dims=all_member_sd_swe.dims,
)


# scf.isel(x=slice(0, 10), y=slice(0, 15))

stacked_data = all_member_sd_swe.data_vars["swe"].stack(n_point=("x", "y")).transpose()
# stacked_data = scf.stack(n_point=("x", "y")).transpose()
# epsilon=1e-3
b_data = np.corrcoef(stacked_data)  # +np.random.randn(*stacked_scf.shape) * epsilon)
# b_data = np.nan_to_num()
b_da = xr.DataArray(
    data=b_data,
    dims=("n_point_i", "n_point_j"),
    coords={
        "n_point_i": ("n_point_i", stacked_data.n_point.values),
        "n_point_j": ("n_point_j", stacked_data.n_point.values),
    },
)

xi_coords = stacked_data.coords["x"].values
yi_coords = stacked_data.coords["y"].values

# Create a MultiIndex for grid_point_i and grid_point_j
grid_points = np.arange(1, len(stacked_data["n_point"]) + 1)

# Assign coordinates to the correlation matrix

cov_matrix = b_da.assign_coords(
    {
        "grid_point_i": ("n_point_i", grid_points),
        "grid_point_j": ("n_point_j", grid_points),
        "xi": ("n_point_i", xi_coords),
        "yi": ("n_point_i", yi_coords),
        "xj": ("n_point_j", xi_coords),  # Same as xi for symmetry
        "yj": ("n_point_j", yi_coords),  # Same as yi for symmetry
    }
)
# Rename dimensions f
cov_matrix = cov_matrix.swap_dims({"n_point_i": "i", "n_point_j": "j"})
cov_matrix.isel(i=slice(500, 600), j=slice(500, 600)).plot.imshow(y="i", x="j", xincrease=False)


observation = xr.open_dataset(f"{folder}/OBSERVATIONS_220226H12.nc")

cloud_mask = np.isnan(observation).rename({"xx": "x", "yy": "y"})
# cmask = cloud_mask.stack(n_point=("x", "y")).transpose()

test_idx = np.random.randint(0, 14443 + 1)

print(test_idx)
idx_corr = cov_matrix.sel(i=test_idx)
idx_corr_nonans = idx_corr.dropna(dim="j", how="all")
k_idx_corr = idx_corr_nonans.sortby(lambda x: x, ascending=False)
# k_idx_corr = k_idx_corr.where(~cmask).isel(j=slice(1,21))
k_idx_corr = k_idx_corr.isel(j=slice(1, 21))

fig, ax = plt.subplots()
all_member_sd_swe.data_vars["swe"].mean(dim="member").where(~cloud_mask.SCF).plot.imshow(ax=ax, cmap="Blues")
ax.plot(idx_corr.xi, idx_corr.yi, "r*")
ax.plot(k_idx_corr.xj, k_idx_corr.yj, "yo")

# sortby(lambda x: x, ascending=False)

# return np.sort(correlations_point_i, axis=1)


kmax = 20
sorted_correlations = cov_matrix.dropna(dim="j", how="all").groupby("i").map(klocalize, (kmax,))
# ax.plot()
# print(k_idx_corr.xj.values,k_idx_corr.yj.values)
