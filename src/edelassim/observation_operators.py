import numpy as np
import xarray as xr


def zaitchik(swe: np.ndarray, tau_scf: int = 4, swe_full_snow_cover: int = 20) -> np.ndarray:
    return np.minimum(1 - (np.exp(-tau_scf * (swe / swe_full_snow_cover)) - (swe / swe_full_snow_cover) * np.exp(-tau_scf)), 1)


def dickinson(sd: np.ndarray | xr.DataArray, b: float = 1.22, a: float = 0.11) -> np.ndarray | xr.DataArray:
    return np.minimum(1, (b * sd) / (sd + a))
