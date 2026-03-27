import numpy as np


def zaitchik(swe: np.ndarray, tau_scf: int = 4, swe_full_snow_cover: int = 20) -> np.ndarray:
    return np.minimum(1 - (np.exp(-tau_scf * (swe / swe_full_snow_cover)) - (swe / swe_full_snow_cover) * np.exp(-tau_scf)), 1)


def dickinson(sd: np.ndarray, a: float, b: float):
    return np.minimum(1, (a * sd) / (sd + b))
