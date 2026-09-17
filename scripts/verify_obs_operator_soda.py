import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from edelassim.observation_operators import dickinson, zaitchik
from edelassim.postprocess_surfex.prep import compute_snow_depth_thickness_mass_from_prep

folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/assim_viirs_local"
for mb in range(1, 18):
    if mb < 10:
        mb_str = f"0{mb}"
    else:
        mb_str = mb
    csv_file = f"{folder}/fsc_soda_0{mb_str}.csv"
    df = pd.read_csv(csv_file, header=None)
    mb_fsc = np.flip(df.values.reshape(101, 143), axis=0)
    break

prep = compute_snow_depth_thickness_mass_from_prep(
    prep_ds=xr.open_dataset(f"{folder}/PREP_220226H12_PF_ENS1.nc"),
    slope_da=xr.open_dataarray(
        "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/250m/SLP_GR_L93_250m.tif"
    ).sel(band=1),
    crs=CRS.from_epsg(2154),
)

# fsc_prep = zaitchik(swe=prep.data_vars["swe"])
fsc_prep = dickinson(sd=prep["snow_thickness"], a=0.11, b=2.1)

# plot y axis is fsc not swe
fig, ax = plt.subplots(figsize=(6, 6))
imshowed = ax.imshow(mb_fsc)
fig.colorbar(imshowed)
fig, ax = plt.subplots(figsize=(6, 6))
imshowed = ax.imshow(fsc_prep)
fig.colorbar(imshowed)
fig, ax = plt.subplots(figsize=(6, 6))
imshowed = ax.imshow(fsc_prep - mb_fsc, cmap="coolwarm")
fig.colorbar(imshowed)

plt.show()
# (fsc_prep - mb_fsc).plot.imshow()
# (fsc_prep * 0 + mb_fsc).to_netcdf("diff_fsc_mb0")
