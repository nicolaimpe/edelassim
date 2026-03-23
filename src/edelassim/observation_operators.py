import numpy as np


def zaitchik(swe: np.ndarray, tau_scf: float, swe_full_snow_cover: float) -> np.ndarray:
    return np.minimum(1 - (np.exp(-tau_scf * (swe / swe_full_snow_cover)) - (swe / swe_full_snow_cover) * np.exp(-tau_scf)), 1)


def dickinson(sd: np.ndarray, a: float, b: float):
    return np.minimum(1, (a * sd) / (sd + b))
