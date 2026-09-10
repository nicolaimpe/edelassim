import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

############ STATIC
COMPASS_ROSE_DICT = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


# Define color stops at specific values
fsc_color_def_snow_cover = [
    (0.0, (0, 0, 0)),  # 0 -> black, no snow
    (1 / 100, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue, 1% snow
    (100 / 100, (1, 1, 1)),  # 100 -> white, full snow
]

FSC_CMAP_SNOW_COVER = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def_snow_cover, N=256)
FSC_CMAP_SNOW_COVER.set_bad("gray")

SNOW_DEPTH_CMAP = plt.get_cmap("Blues")
SNOW_DEPTH_CMAP.set_under("black")
SNOW_DEPTH_CMAP.set_bad("gray")

FIELD_DIFF_CMAP = plt.get_cmap("coolwarm_r")
FIELD_DIFF_CMAP.set_bad("gray")

PRECIP_CMAP = plt.get_cmap("viridis")
PRECIP_CMAP.set_bad("gray")

PHASE_CMAP = plt.get_cmap("Blues_r")
PHASE_CMAP.set_bad("gray")


def add_colorbar(ax, **kwargs):
    """Add a colorbar to the given axes, safely removing any existing first."""
    # Safely remove existing colorbar and its axes
    if hasattr(ax, "_colorbar") and ax._colorbar is not None:
        try:
            ax._colorbar.remove()
        except (AttributeError, ValueError, KeyError):
            pass  # Already removed or invalid

    if hasattr(ax, "_colorbar_ax") and ax._colorbar_ax is not None:
        try:
            ax._colorbar_ax.remove()
        except (AttributeError, ValueError, KeyError):
            pass

    if ax.images:
        mappable = ax.images[-1]
    elif ax.collections:
        mappable = ax.collections[-1]
    else:
        raise ValueError("No mappable found in axes")

    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, **kwargs)

    # Store references
    ax._colorbar = cb
    ax._colorbar_ax = cax
    return cb


def plot_elevation_lines(ax: Axes, dem: xr.DataArray, elevation_step: int = 300):
    elevation_min = np.floor(dem.min() / elevation_step) * elevation_step
    elevation_max = np.ceil(dem.max() / elevation_step) * elevation_step
    levels = np.arange(elevation_min, elevation_max, elevation_step)
    elevation_lines = ax.contour(dem.values, levels=levels, colors="black", linewidths=0.2)
    ax.clabel(elevation_lines, levels, inline=True, fontsize=5, fmt="%.1f")


def add_2d_plot(data: np.ndarray, ax: Axes, dem: xr.DataArray, title: str | None = None, **kwargs):
    ax.imshow(data, **kwargs)
    add_colorbar(ax=ax)
    plot_elevation_lines(ax=ax, dem=dem)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
