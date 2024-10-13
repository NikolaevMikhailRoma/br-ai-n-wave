from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

class SegPlot:
    def __init__(self):
        self.fig = None
        self.ax = None

    def plot(self, data: np.ndarray, ax: Optional[plt.Axes] = None, cmap: str = 'seismic',
             aspect: str = 'auto', vmin: Optional[float] = None, vmax: Optional[float] = None) -> AxesImage:
        if ax is None:
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots()
            ax = self.ax

        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        im = ax.imshow(data, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)
        return im