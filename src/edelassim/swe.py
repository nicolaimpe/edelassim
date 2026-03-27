import numpy as np


def sd_to_swe(sd: np.ndarray, snow_density: float = 320) -> np.ndarray:
    """SD meters, SWE mm
    mean_snow_density = 320  # kg/m^3"""
    water_density = 1000  # kg/m^3
    m_to_mm_scale = 1000
    return sd * snow_density / water_density * m_to_mm_scale


def swe_to_sd(swe: np.ndarray, snow_density: float = 320) -> np.ndarray:
    """SD meters, SWE mm
    mean_snow_density = 320  # kg/m^3"""
    water_density = 1000  # kg/m^3
    m_to_mm_scale = 1000
    return swe / snow_density * water_density / m_to_mm_scale
