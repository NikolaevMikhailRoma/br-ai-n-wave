import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from typing import Tuple
from SegReader import SeqReader
import os
from config import DATA_DIR
from SegPlot import SegPlot

class SegInteractPlot:
    def __init__(self, seq_reader, df, stats: dict):
        self.seq_reader = seq_reader
        self.df = df
        self.stats = stats
        self.current_mode = 'inline'
        self.current_slice = self.seq_reader.get_dimensions()['INLINE_3D'][0]
        self.use_three_sigma = True
        self.use_color = True
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.seg_plot = SegPlot()
        self.setup_plot()

    def setup_plot(self):
        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.95)

        # Setup mode selection
        rax = plt.axes((0.1, 0.1, 0.13, 0.13))
        self.radio = RadioButtons(rax, ('inline', 'crossline', 'depth'))
        self.radio.on_clicked(self.mode_changed)

        # Setup slider
        self.slider_ax = plt.axes((0.1, 0.05, 0.8, 0.03))
        vmin, vmax = self.seq_reader.get_dimensions()['INLINE_3D']
        self.slider = Slider(self.slider_ax, 'Slice', vmin, vmax, valinit=vmin, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Setup color scale toggle button
        self.scale_button_ax = plt.axes((0.25, 0.15, 0.15, 0.05))
        self.scale_button = Button(self.scale_button_ax, 'Toggle Scale')
        self.scale_button.on_clicked(self.toggle_color_scale)

        # Setup color mode toggle button
        self.color_button_ax = plt.axes((0.45, 0.15, 0.15, 0.05))
        self.color_button = Button(self.color_button_ax, 'Toggle Color')
        self.color_button.on_clicked(self.toggle_color_mode)

        # Setup save button
        self.save_button_ax = plt.axes((0.65, 0.15, 0.15, 0.05))
        self.save_button = Button(self.save_button_ax, 'Save Image')
        self.save_button.on_clicked(self.save_image)

        # Setup manual slice input
        self.slice_input_ax = plt.axes((0.25, 0.22, 0.1, 0.05))
        self.slice_input = TextBox(self.slice_input_ax, 'Slice:', initial=str(int(vmin)))
        self.slice_input.on_submit(self.manual_slice_update)

        # Display slice range
        self.slice_range_ax = plt.axes((0.45, 0.22, 0.3, 0.05))
        self.slice_range_ax.axis('off')
        self.slice_range_text = self.slice_range_ax.text(0, 0, f'Range: {int(vmin)} - {int(vmax)}')

        # Initialize colorbar and image
        self.colorbar = None
        self.im = None

        self.update_plot()

    def mode_changed(self, label):
        self.current_mode = label
        self.update_slider_range()
        self.update_plot()

    def update_slider_range(self):
        if self.current_mode == 'inline':
            vmin, vmax = self.seq_reader.get_dimensions()['INLINE_3D']
        elif self.current_mode == 'crossline':
            vmin, vmax = self.seq_reader.get_dimensions()['CROSSLINE_3D']
        else:  # depth
            vmin, vmax = 0, self.seq_reader.get_dimensions()['n_samples'] - 1

        self.slider.valmin = vmin
        self.slider.valmax = vmax
        self.slider.set_val(vmin)
        self.slider_ax.set_xlim(vmin, vmax)
        self.slice_range_text.set_text(f'Range: {int(vmin)} - {int(vmax)}')
        self.slice_input.set_val(str(int(vmin)))

    def update_slice(self, val):
        self.current_slice = int(val)
        self.slice_input.set_val(str(self.current_slice))
        self.update_plot()

    def manual_slice_update(self, text):
        try:
            new_slice = int(text)
            if self.slider.valmin <= new_slice <= self.slider.valmax:
                self.current_slice = new_slice
                self.slider.set_val(new_slice)
                self.update_plot()
            else:
                print(f"Slice value out of range. Please enter a value between {int(self.slider.valmin)} and {int(self.slider.valmax)}.")
        except ValueError:
            print("Please enter a valid integer.")

    def get_slice_data(self) -> Tuple[np.ndarray, str]:
        if self.current_mode == 'inline':
            data = self.seq_reader.get_inline_slice(self.current_slice)
            title = f'Inline {self.current_slice}'
        elif self.current_mode == 'crossline':
            data = self.seq_reader.get_crossline_slice(self.current_slice)
            title = f'Crossline {self.current_slice}'
        else:  # depth
            data = self.seq_reader.get_depth_slice(self.current_slice)
            title = f'Depth Slice {self.current_slice}'
        return data, title

    def get_color_scale(self):
        if self.use_three_sigma:
            vmin = max(self.stats['sigma_minus_3'], self.stats['min'])
            vmax = min(self.stats['sigma_3'], self.stats['max'])
        else:
            vmin = self.stats['min']
            vmax = self.stats['max']
        return vmin, vmax

    def toggle_color_scale(self, event):
        self.use_three_sigma = not self.use_three_sigma
        self.update_plot()

    def toggle_color_mode(self, event):
        self.use_color = not self.use_color
        self.update_plot()

    def save_image(self, event):
        data, _ = self.get_slice_data()
        images_dir = os.path.join(DATA_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Save as PNG
        filename = f"{self.current_mode}_{self.current_slice}.png"
        filepath = os.path.join(images_dir, filename)
        plt.imsave(filepath, data, cmap='seismic' if self.use_color else 'gray')

        # Save as NPY
        npy_filename = f"{self.current_mode}_{self.current_slice}.npy"
        npy_filepath = os.path.join(images_dir, npy_filename)
        np.save(npy_filepath, data)

        print(f"Image saved as {filepath} and {npy_filepath}")

    def update_plot(self):
        self.ax.clear()
        data, title = self.get_slice_data()
        vmin, vmax = self.get_color_scale()
        cmap = 'seismic' if self.use_color else 'gray'

        self.im = self.seg_plot.plot(data, self.ax, cmap, 'auto', vmin, vmax)

        self.ax.set_title(title)
        self.ax.set_xlabel('Trace Number')
        self.ax.set_ylabel('Time/Depth')

        # Update or create colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax)
            self.colorbar.set_label('Amplitude')
        else:
            self.colorbar.update_normal(self.im)

        scale_type = "Three-sigma" if self.use_three_sigma else "Min-Max"
        self.scale_button.label.set_text(f'Scale: {scale_type}')

        color_mode = "Color" if self.use_color else "Grayscale"
        self.color_button.label.set_text(f'Mode: {color_mode}')

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == '__main__':
    from config import SEGFAST_FILE_PATH

    reader = SeqReader(SEGFAST_FILE_PATH)
    stats = reader.get_statistics()
    df = reader.get_coordinates()

    # Создание интерактивного визуализатора
    interactive_plot = SegInteractPlot(reader, df, stats)
    interactive_plot.show()
