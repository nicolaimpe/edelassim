from typing import Tuple

import numpy as np
import xarray as xr


def find_common_correspondences(data_1: xr.Dataset, data_2: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    data_1_valid_mask = ~np.isnan(data_1)
    data_2_valid_mask = ~np.isnan(data_2)

    union = data_1_valid_mask * data_2_valid_mask

    data_1_correspondences = data_1.where(union).values.flatten()
    data_2_correspondences = data_2.where(union).values.flatten()

    data_1_correspondences = data_1_correspondences[~np.isnan(data_1_correspondences)]
    data_2_correspondences = data_2_correspondences[~np.isnan(data_2_correspondences)]
    return data_1_correspondences, data_2_correspondences
