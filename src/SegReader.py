import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from seismiqb import Field

class SeqReader:
    """
    A class for reading and analyzing SEG-Y seismic data files using seismiqb.
    
    This class provides functionality to load SEG-Y files, compute statistics,
    generate histograms, and extract various slices of seismic data.
    
    Attributes:
        file_path (str): Path to the SEG-Y file.
        field (Field): seismiqb Field instance for data access
        df (pd.DataFrame): DataFrame containing trace headers
        broken_traces (List[int]): List of indices of broken traces
    """
    
    def __init__(self, file_path: str, engine: str = None):
        """
        Initialize the SeqReader object.
        
        Args:
            file_path (str): Path to the SEG-Y file
            engine (str, optional): Not used, kept for backward compatibility
        """
        self.file_path = file_path
        self.broken_traces = []
        
        # Initialize field if metadata doesn't exist
        if not os.path.exists(file_path + '_meta'):
            Field(file_path, geometry_kwargs={'loader_class': 'segyio'})
            
        # Load field and headers
        self.field = Field(file_path)
        self.df = self.field.geometry.headers
        
        # Cache statistics
        self._statistics = self._calculate_statistics()
        self._dimensions = self._calculate_dimensions()

    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistical parameters from seismic data."""
        mean = float(self.field.mean)
        std_dev = float(self.field.std)
        
        return {
            'min': float(self.field.min),
            'max': float(self.field.max),
            'mean': mean,
            'median': float(self.field.get_quantile(0.5)),
            'variance': float(std_dev ** 2),
            'std_dev': std_dev,
            'q1': float(self.field.get_quantile(0.25)),
            'q3': float(self.field.get_quantile(0.75)),
            'sigma_1': mean + std_dev,
            'sigma_2': mean + 2 * std_dev,
            'sigma_3': mean + 3 * std_dev,
            'sigma_minus_1': mean - std_dev,
            'sigma_minus_2': mean - 2 * std_dev,
            'sigma_minus_3': mean - 3 * std_dev
        }

    def _calculate_dimensions(self) -> Dict[str, Tuple[int, int]]:
        """Calculate dimensions of the seismic data."""
        shape = self.field.geometry.shape
        return {
            'INLINE_3D': (int(self.df['INLINE_3D'].min()), int(self.df['INLINE_3D'].max())),
            'CROSSLINE_3D': (int(self.df['CROSSLINE_3D'].min()), int(self.df['CROSSLINE_3D'].max())),
            'n_samples': int(shape[2])  # Assuming last dimension is samples
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical parameters of the seismic data."""
        return self._statistics

    def plot_histogram(self) -> None:
        """Plot a histogram of the seismic data with statistical indicators."""
        import matplotlib.pyplot as plt
        
        stats = self.get_statistics()
        # Get data for histogram using crop
        locations = [slice(0, 100), slice(0, 100), slice(0, 100)]  # Adjust size as needed
        data = self.field.geometry.load_crop(locations).flatten()
        
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(data, bins=100, density=True, alpha=0.7)
        
        plt.axvline(stats['mean'], color='r', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
        plt.axvline(stats['median'], color='g', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}")
        plt.axvline(stats['q1'], color='c', linestyle='dotted', linewidth=2, label=f"Q1: {stats['q1']:.2f}")
        plt.axvline(stats['q3'], color='m', linestyle='dotted', linewidth=2, label=f"Q3: {stats['q3']:.2f}")
        
        colors = ['y', 'orange', 'r']
        for i, sigma in enumerate([1, 2, 3]):
            plt.axvline(stats[f'sigma_{sigma}'], color=colors[i], linestyle=':', linewidth=2,
                       label=f"+{sigma}σ: {stats[f'sigma_{sigma}']:.2f}")
            plt.axvline(stats[f'sigma_minus_{sigma}'], color=colors[i], linestyle=':', linewidth=2,
                       label=f"-{sigma}σ: {stats[f'sigma_minus_{sigma}']:.2f}")
        
        plt.title("Histogram of Seismic Data with 3-Sigma Rule")
        plt.xlabel("Amplitude")
        plt.ylabel("Density")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.text(0.05, 0.95, f"Std Dev: {stats['std_dev']:.2f}", transform=plt.gca().transAxes, verticalalignment='top')
        plt.tight_layout()
        plt.show()

    def get_inline_slice(self, inline_index: int) -> np.ndarray:
        """Get a 2D slice along the inline direction."""
        # Find the index in the geometry that corresponds to the inline number
        inline_position = np.where(np.unique(self.df['INLINE_3D']) == inline_index)[0][0]
        # Load and transpose to match original orientation
        slice_data = self.field.geometry.load_slide(index=inline_position, axis=0)
        return slice_data.T

    def get_crossline_slice(self, crossline_index: int) -> np.ndarray:
        """Get a 2D slice along the crossline direction."""
        crossline_position = np.where(np.unique(self.df['CROSSLINE_3D']) == crossline_index)[0][0]
        # Load and transpose to match original orientation
        slice_data = self.field.geometry.load_slide(index=crossline_position, axis=1)
        return slice_data.T

    def get_depth_slice(self, depth_index: int) -> np.ndarray:
        """Get a 2D slice at a specific depth."""
        return self.field.geometry.load_slide(index=depth_index, axis=2)

    def get_dimensions(self) -> Dict[str, Tuple[int, int]]:
        """Get the dimensions of the seismic data."""
        return self._dimensions

    def get_trace_shape(self) -> Tuple[int, int]:
        """Get the shape of a single trace."""
        shape = self.field.geometry.shape
        return (shape[0], shape[1])  # inline, crossline dimensions

    def get_broken_traces(self) -> List[int]:
        """Get the list of indices of broken traces."""
        return self.broken_traces

    def get_coordinates(self) -> pd.DataFrame:
        """Get the coordinates of all traces."""
        return self.df[['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']]


if __name__ == '__main__':
    from config import SEGFAST_FILE_PATH
    
    reader = SeqReader(SEGFAST_FILE_PATH)
    
    # Test basic functionality
    stats = reader.get_statistics()
    print("Statistics:", stats)
    
    reader.plot_histogram()
    
    # Test slices
    inline_slice = reader.get_inline_slice(reader.get_dimensions()['INLINE_3D'][0])
    print("Inline slice shape:", inline_slice.shape)
    
    dimensions = reader.get_dimensions()
    print("Dimensions:", dimensions)