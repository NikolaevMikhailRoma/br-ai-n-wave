import numpy as np
import pyvista as pv
from SegReader import SeqReader


class PyVistaSeg3DVisualization:
    def __init__(self, seq_reader: SeqReader):
        self.seq_reader = seq_reader
        self.dimensions = seq_reader.get_dimensions()
        self.stats = seq_reader.get_statistics()

        self.inline_index = self.dimensions['INLINE_3D'][0]
        self.crossline_index = self.dimensions['CROSSLINE_3D'][0]
        self.depth_index = self.dimensions['n_samples'] - 1  # Начинаем с самого нижнего среза

        self.plotter = pv.Plotter(window_size=[1024, 768])  # Увеличиваем размер окна
        self.setup_plot()

    def setup_plot(self):
        # Создание и добавление всех трех слайсов
        self.create_inline_slice()
        self.create_crossline_slice()
        self.create_depth_slice()

        # Настройка цветового масштабирования
        self.vmin = max(self.stats['sigma_minus_3'], self.stats['min'])
        self.vmax = min(self.stats['sigma_3'], self.stats['max'])

        # Добавление слайсов на плоттер
        self.inline_surf = self.plotter.add_mesh(self.inline_grid, scalars="values", cmap="seismic",
                                                 clim=[self.vmin, self.vmax], show_scalar_bar=True,
                                                 scalar_bar_args={"title": "Amplitude"})
        self.crossline_surf = self.plotter.add_mesh(self.crossline_grid, scalars="values", cmap="seismic",
                                                    clim=[self.vmin, self.vmax], show_scalar_bar=False)
        self.depth_surf = self.plotter.add_mesh(self.depth_grid, scalars="values", cmap="seismic",
                                                clim=[self.vmin, self.vmax], show_scalar_bar=False)

        # Добавление ограничивающего параллелепипеда (box) и осей
        self.add_bounding_box()

        # Настройка камеры
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        # Добавление слайдеров с разным позиционированием
        self.plotter.add_slider_widget(
            callback=self.update_inline_slice,
            rng=[self.dimensions['INLINE_3D'][0], self.dimensions['INLINE_3D'][1]],
            value=self.inline_index,
            title="Inline",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style='modern'
        )
        self.plotter.add_slider_widget(
            callback=self.update_crossline_slice,
            rng=[self.dimensions['CROSSLINE_3D'][0], self.dimensions['CROSSLINE_3D'][1]],
            value=self.crossline_index,
            title="Crossline",
            pointa=(0.025, 0.06),
            pointb=(0.31, 0.06),
            style='modern'
        )
        self.plotter.add_slider_widget(
            callback=self.update_depth_slice,
            rng=[0, self.dimensions['n_samples'] - 1],
            value=self.depth_index,
            title="Depth",
            pointa=(0.025, 0.02),
            pointb=(0.31, 0.02),
            style='modern'
        )

    def create_inline_slice(self):
        inline_slice = self.seq_reader.get_inline_slice(self.inline_index)
        x = np.arange(self.dimensions['CROSSLINE_3D'][1] - self.dimensions['CROSSLINE_3D'][0] + 1)
        z = np.arange(self.dimensions['n_samples'])[::-1]
        xx, zz = np.meshgrid(x, z)
        yy = np.full_like(xx, self.inline_index - self.dimensions['INLINE_3D'][0])
        self.inline_grid = pv.StructuredGrid(xx, yy, zz)
        self.inline_grid.point_data["values"] = inline_slice.ravel(order="F")

    def create_crossline_slice(self):
        crossline_slice = self.seq_reader.get_crossline_slice(self.crossline_index)
        y = np.arange(self.dimensions['INLINE_3D'][1] - self.dimensions['INLINE_3D'][0] + 1)
        z = np.arange(self.dimensions['n_samples'])[::-1]
        yy, zz = np.meshgrid(y, z)
        xx = np.full_like(yy, self.crossline_index - self.dimensions['CROSSLINE_3D'][0])
        self.crossline_grid = pv.StructuredGrid(xx, yy, zz)
        self.crossline_grid.point_data["values"] = crossline_slice.ravel(order="F")

    def create_depth_slice(self):
        depth_slice = self.seq_reader.get_depth_slice(self.depth_index)
        x = np.arange(self.dimensions['CROSSLINE_3D'][1] - self.dimensions['CROSSLINE_3D'][0] + 1)
        y = np.arange(self.dimensions['INLINE_3D'][1] - self.dimensions['INLINE_3D'][0] + 1)
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, self.dimensions['n_samples'] - 1 - self.depth_index)
        self.depth_grid = pv.StructuredGrid(xx, yy, zz)
        self.depth_grid.point_data["values"] = depth_slice.ravel(order="F")

    def add_bounding_box(self):
        # Создаем ограничивающий параллелепипед (box)
        box = pv.Box(bounds=(0, self.dimensions['CROSSLINE_3D'][1] - self.dimensions['CROSSLINE_3D'][0],
                             0, self.dimensions['INLINE_3D'][1] - self.dimensions['INLINE_3D'][0],
                             0, self.dimensions['n_samples'] - 1))

        # Добавляем box на плоттер
        self.plotter.add_mesh(box, color="black", style="wireframe", line_width=2, opacity=0.5)

        # Добавляем оси с пометками
        # self.plotter.add_axes(xlabel='Crossline', ylabel='Inline', zlabel='Time/Depth',
        #                       x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True,
        #                       line_width=5, labels_off=False)

        # Добавляем пометки на оси
        crossline_range = self.dimensions['CROSSLINE_3D'][1] - self.dimensions['CROSSLINE_3D'][0]
        inline_range = self.dimensions['INLINE_3D'][1] - self.dimensions['INLINE_3D'][0]
        depth_range = self.dimensions['n_samples'] - 1

        # self.plotter.add_axis_labels(
        #     x_labels=[(0, str(self.dimensions['CROSSLINE_3D'][0])),
        #               (crossline_range, str(self.dimensions['CROSSLINE_3D'][1]))],
        #     y_labels=[(0, str(self.dimensions['INLINE_3D'][0])),
        #               (inline_range, str(self.dimensions['INLINE_3D'][1]))],
        #     z_labels=[(0, str(depth_range)),
        #               (depth_range, '0')]
        # )

    def update_inline_slice(self, value):
        self.inline_index = int(value)
        inline_slice = self.seq_reader.get_inline_slice(self.inline_index)
        self.inline_grid.points[:, 1] = self.inline_index - self.dimensions['INLINE_3D'][0]
        self.inline_grid.point_data["values"] = inline_slice.ravel(order="F")
        self.inline_surf.SetMapper(None)
        self.inline_surf = self.plotter.add_mesh(self.inline_grid, scalars="values", cmap="seismic",
                                                 clim=[self.vmin, self.vmax], show_scalar_bar=True,
                                                 scalar_bar_args={"title": "Amplitude"})
        self.plotter.render()

    def update_crossline_slice(self, value):
        self.crossline_index = int(value)
        crossline_slice = self.seq_reader.get_crossline_slice(self.crossline_index)
        self.crossline_grid.points[:, 0] = self.crossline_index - self.dimensions['CROSSLINE_3D'][0]
        self.crossline_grid.point_data["values"] = crossline_slice.ravel(order="F")
        self.crossline_surf.SetMapper(None)
        self.crossline_surf = self.plotter.add_mesh(self.crossline_grid, scalars="values", cmap="seismic",
                                                    clim=[self.vmin, self.vmax], show_scalar_bar=False)
        self.plotter.render()

    def update_depth_slice(self, value):
        self.depth_index = int(value)
        depth_slice = self.seq_reader.get_depth_slice(self.depth_index)
        self.depth_grid.points[:, 2] = self.dimensions['n_samples'] - 1 - self.depth_index
        self.depth_grid.point_data["values"] = depth_slice.ravel(order="F")
        self.depth_surf.SetMapper(None)
        self.depth_surf = self.plotter.add_mesh(self.depth_grid, scalars="values", cmap="seismic",
                                                clim=[self.vmin, self.vmax], show_scalar_bar=False)
        self.plotter.render()

    def show(self):
        self.plotter.show()


# Пример использования
if __name__ == '__main__':
    from config import SEGFAST_FILE_PATH

    reader = SeqReader(SEGFAST_FILE_PATH)
    viz = PyVistaSeg3DVisualization(reader)
    viz.show()