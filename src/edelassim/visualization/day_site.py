from datetime import timedelta

import matplotlib
import numpy as np
from matplotlib import colors


def RdGn_colormap():
    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Greens"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))

    return colors.ListedColormap(newcolors, name="RdGn")
