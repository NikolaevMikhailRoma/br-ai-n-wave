import os
import platform
import numpy as np
import pandas as pd
import segfast
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import hashlib


class SeqReader:
    """
    A class for reading and analyzing SEG-Y seismic data files.

    This class provides functionality to load SEG-Y files, compute statistics,
    generate histograms, and extract various slices of seismic data. It also
    implements a caching mechanism to improve performance on subsequent runs.

    Attributes:
        file_path (str): Path to the SEG-Y file.
        engine (str): The engine used to read the SEG-Y file ('segyio' or 'memmap').
        metadata_file (str): Path to the metadata cache file.
        segfast_file (segfast.File): Loaded SEG-Y file object.
        df (pd.DataFrame): DataFrame containing trace headers.
        broken_traces (List[int]): List of indices of broken traces.
        _statistics (Dict[str, float]): Cached statistical data.
        _dimensions (Dict[str, Tuple[int, int]]): Cached dimension data.
    """

    def __init__(self, file_path: str, engine: str = None):
        """
        Initialize the SeqReader object.

        Args:
            file_path (str): Path to the SEG-Y file.
            engine (str, optional): The engine to use for reading the SEG-Y file.
                If None, it will be automatically determined based on the platform.
        """
        self.file_path: str = file_path
        self.engine: str = engine or ('segyio' if platform.system() == 'Darwin' else 'memmap')
        self.metadata_file: str = self._get_metadata_file_path()
        self.segfast_file: Union[segfast.File, None] = None
        self.df: Union[pd.DataFrame, None] = None
        self.broken_traces: List[int] = []
        self._statistics: Union[Dict[str, float], None] = None
        self._dimensions: Union[Dict[str, Tuple[int, int]], None] = None

        self._load_or_create_metadata()

    def _get_metadata_file_path(self) -> str:
        """
        Generate a unique metadata file path based on the input file's path and modification time.

        Returns:
            str: Path to the metadata file.
        """
        file_stats = os.stat(self.file_path)
        unique_id = f"{self.file_path}_{file_stats.st_mtime}"
        filename_hash = hashlib.md5(unique_id.encode()).hexdigest()
        return os.path.join(os.path.dirname(self.file_path), f"{filename_hash}_metadata.json")

    def _load_or_create_metadata(self) -> None:
        """
        Load existing metadata if available, otherwise create new metadata.
        """
        if os.path.exists(self.metadata_file):
            self._load_metadata()
        else:
            self._create_metadata()

    def _load_metadata(self) -> None:
        """
        Load metadata from the cache file.
        """
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        self._statistics = metadata['statistics']
        self._dimensions = metadata['dimensions']
        self.broken_traces = metadata['broken_traces']
        self.df = pd.DataFrame(metadata['headers'])

    def _create_metadata(self) -> None:
        """
        Create metadata by processing the SEG-Y file and save it to the cache file.
        """
        self.segfast_file = self._load_segfast_file()
        self.df = self._load_headers()
        self._calculate_statistics()
        self._calculate_dimensions()

        metadata = {
            'statistics': self._statistics,
            'dimensions': self._dimensions,
            'broken_traces': self.broken_traces,
            'headers': self.df.to_dict(orient='list')
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def _load_segfast_file(self) -> segfast.File:
        """
        Load the SEG-Y file using the segfast library.

        Returns:
            segfast.File: Loaded SEG-Y file object.

        Raises:
            FileNotFoundError: If the specified SEG-Y file does not exist.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        return segfast.open(self.file_path, engine=self.engine)

    def _load_headers(self) -> pd.DataFrame:
        """
        Load trace headers from the SEG-Y file.

        Returns:
            pd.DataFrame: DataFrame containing trace headers.
        """
        df = self.segfast_file.load_headers(['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y'])
        # df.columns = ['TRACE_SEQUENCE_FILE', 'INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']
        return df

    def _load_traces(self, traces_to_load: List[int]) -> np.ndarray:
        """
        Load specified traces from the SEG-Y file.

        Args:
            traces_to_load (List[int]): List of trace indices to load.

        Returns:
            np.ndarray: Array of loaded traces.
        """
        if self.segfast_file is None:
            self.segfast_file = self._load_segfast_file()

        loaded_traces = []
        for trace in traces_to_load:
            try:
                loaded_trace = self.segfast_file.load_traces([trace], buffer=None)
                loaded_traces.append(loaded_trace[0])
            except OSError:
                self.broken_traces.append(trace)
        # return np.array(loaded_traces)
        return np.array(loaded_traces).T

    def _calculate_statistics(self) -> None:
        """
        Calculate statistical parameters from all seismic traces.
        """
        data = []
        for i in range(len(self.df)):
            trace = self._load_traces([i])
            if len(trace) > 0:
                data.extend(trace.flatten())

        data = np.array(data)
        mean = float(np.mean(data))
        std_dev = float(np.std(data))
        self._statistics = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': mean,
            'median': float(np.median(data)),
            'variance': float(np.var(data)),
            'std_dev': std_dev,
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'sigma_1': mean + std_dev,
            'sigma_2': mean + 2 * std_dev,
            'sigma_3': mean + 3 * std_dev,
            'sigma_minus_1': mean - std_dev,
            'sigma_minus_2': mean - 2 * std_dev,
            'sigma_minus_3': mean - 3 * std_dev
        }

    def _calculate_dimensions(self) -> None:
        """
        Calculate dimensions of the seismic data.
        """
        self._dimensions = {
            'INLINE_3D': (int(self.df['INLINE_3D'].min()), int(self.df['INLINE_3D'].max())),
            'CROSSLINE_3D': (int(self.df['CROSSLINE_3D'].min()), int(self.df['CROSSLINE_3D'].max())),
            'n_samples': int(self.segfast_file.n_samples)
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical parameters of the seismic data.

        Returns:
            Dict[str, float]: Dictionary containing various statistical measures.
        """
        return self._statistics

    def plot_histogram(self) -> None:
        """
        Plot a histogram of the seismic data with statistical indicators.

        This method generates a histogram of all seismic traces and overlays
        various statistical measures including mean, median, quartiles, and
        indicators for the three-sigma rule.
        """
        stats = self.get_statistics()
        data = []
        for i in range(len(self.df)):
            trace = self._load_traces([i])
            if len(trace) > 0:
                data.extend(trace.flatten())

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
        """
        Get a 2D slice along the inline direction.

        Args:
            inline_index (int): Index of the inline to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the inline slice.
        """
        traces_to_load = self.df[self.df['INLINE_3D'] == inline_index]['TRACE_SEQUENCE_FILE'].tolist()
        return self._load_traces(traces_to_load)

    def get_crossline_slice(self, crossline_index: int) -> np.ndarray:
        """
        Get a 2D slice along the crossline direction.

        Args:
            crossline_index (int): Index of the crossline to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the crossline slice.
        """
        traces_to_load = self.df[self.df['CROSSLINE_3D'] == crossline_index]['TRACE_SEQUENCE_FILE'].tolist()
        return self._load_traces(traces_to_load)

    def get_depth_slice(self, depth_index: int) -> np.ndarray:
        """
        Get a 2D slice at a specific depth.

        Args:
            depth_index (int): Index of the depth to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the depth slice.
        """
        if self.segfast_file is None:
            self.segfast_file = self._load_segfast_file()
        depth_slice = self.segfast_file.load_depth_slices([depth_index], buffer=None)[0]
        return depth_slice.reshape(self.df['INLINE_3D'].nunique(), self.df['CROSSLINE_3D'].nunique())

    def get_dimensions(self) -> Dict[str, Tuple[int, int]]:
        """
        Get the dimensions of the seismic data.

        Returns:
            Dict[str, Tuple[int, int]]: Dictionary containing the range of inline and crossline indices,
                                        and the number of samples per trace.
        """
        return self._dimensions

    def get_trace_shape(self) -> Tuple[int, int]:
        """
        Get the shape of a single trace.

        Returns:
            Tuple[int, int]: Shape of a single trace (number of inlines, number of crosslines).
        """
        return (self.df['INLINE_3D'].nunique(), self.df['CROSSLINE_3D'].nunique())

    def get_broken_traces(self) -> List[int]:
        """
        Get the list of indices of broken traces.

        Returns:
            List[int]: List of indices of broken traces.
        """
        return self.broken_traces

    def get_coordinates(self) -> pd.DataFrame:
        """
        Get the coordinates of all traces.

        Returns:
            pd.DataFrame: DataFrame containing INLINE_3D, CROSSLINE_3D, CDP_X, and CDP_Y coordinates.
        """
        return self.df[['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']]


if __name__ == '__main__':
    from config import SEGFAST_FILE_PATH

    reader = SeqReader(SEGFAST_FILE_PATH)

    stats = reader.get_statistics()
    reader.plot_histogram()
    inline_slice = reader.get_inline_slice(100)
    dimensions = reader.get_dimensions()
