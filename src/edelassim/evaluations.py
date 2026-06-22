from typing import Tuple

import numpy as np
import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS


class GrandesRoussesGrid20m(GSGrid):
    """This grid bound correspond to a bounding box including all mountaineous areas over metropolitan France in WGS84 geographic coordinates."""

    def __init__(self):
        super().__init__(
            x0=736914.0445886956,
            y0=5015138.075400733,
            resolution=20,
            width=1788,
            height=1263,
            crs=CRS.from_epsg(32631),
            name="Sentinel2_GrandesRousses",
        )


def find_common_correspondences(data_1: xr.Dataset, data_2: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    data_1_valid_mask = ~np.isnan(data_1)
    data_2_valid_mask = ~np.isnan(data_2)

    union = data_1_valid_mask * data_2_valid_mask

    data_1_correspondences = data_1.where(union).values.flatten()
    data_2_correspondences = data_2.where(union).values.flatten()

    data_1_correspondences = data_1_correspondences[~np.isnan(data_1_correspondences)]
    data_2_correspondences = data_2_correspondences[~np.isnan(data_2_correspondences)]
    return data_1_correspondences, data_2_correspondences
