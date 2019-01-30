import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.image import AxesImage
from matplotlib.gridspec import GridSpec
from theory.represent import from_WH_to_rank_1_list


def plot_factorisation_simple(W, H, grid_shape=None):
    l = from_WH_to_rank_1_list(W, H)
    if grid_shape is None:
        s = np.sqrt(len(l))
        grid_shape = (np.ceil(s), np.floor(s))
        if grid_shape[0] * grid_shape[1] < len(l):
            grid_shape = (np.ceil(s), np.ceil(s))

    plt.figure()
    for i, mat in enumerate(l):
        i = i + 1
        plt.subplot(grid_shape[0], grid_shape[1], i)
        plt.imshow(mat)
        plt.title(str(i))


class InteractiveFactorPlot:
    def __init__(self, W, H, V):
        gs = GridSpec(2, 3, width_ratios=[1, 10, 1], height_ratios=[10, 1])

        self.fact_fig = plt.figure(figsize=(5, 4))
        self.fact_ax = plt.subplot(gs[0, :])
        self.btn_less_ax = plt.subplot(gs[1, 0])
        self.slider_ax = plt.subplot(gs[1, 1])
        self.btn_more_ax = plt.subplot(gs[1, 2])
        self.fact_fig.suptitle("Rank 1 terms")

        self.ori_fig = plt.figure(figsize=(10, 4))
        self.ori_ax = plt.subplot(121)
        self.ori_ax.set_title("Original matrix V")
        self.wh_ax = plt.subplot(122)
        self.wh_ax.set_title("W @ H")

        self.WH = W @ H
        self.V = V
        self.terms = from_WH_to_rank_1_list(W, H)
        self.term_plot_range = [0, max(np.max(t) for t in self.terms)]
        self.result_plot_range = [0, max(np.max(self.WH), np.max(self.V))]

        self.slider = widgets.Slider(self.slider_ax,
                                     label="",
                                     valmin=0,
                                     valmax=len(self.terms) - 1,
                                     valfmt='%0.0f',
                                     valstep=1)
        self.slider.on_changed(self.on_slider_move)

        self.btn_less = widgets.Button(self.btn_less_ax, "<<")
        self.btn_less.on_clicked(lambda e: self.set_slider_val(self.slider.val - 1))

        self.btn_more = widgets.Button(self.btn_more_ax, ">>")
        self.btn_more.on_clicked(lambda e: self.set_slider_val(self.slider.val + 1))

        self.ori_ax.imshow(V, picker=True, zorder=0,
                           vmin=self.result_plot_range[0],
                           vmax=self.result_plot_range[1])
        self.wh_ax.imshow(self.WH, picker=True, zorder=0,
                           vmin=self.result_plot_range[0],
                           vmax=self.result_plot_range[1])
        self.ori_fig.canvas.mpl_connect('pick_event', self.on_image_clicked)

        self.selected_sorting_points = []

        self.on_slider_move(0)

    def set_slider_val(self, val):
        val = np.clip(val, self.slider.valmin, self.slider.valmax)
        self.slider.set_val(val)

    def on_slider_move(self, val):
        self.plot_ith_term(int(val))

    def plot_ith_term(self, i):
        for artist in self.fact_ax.get_children():
            if isinstance(artist, AxesImage): artist.remove()

        self.fact_ax.imshow(self.terms[i], zorder=0, vmin=self.term_plot_range[0], vmax=self.term_plot_range[1])
        self.fact_fig.canvas.draw()

    def on_image_clicked(self, event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            coords = (int(np.rint(event.mouseevent.xdata)),
                      int(np.rint(event.mouseevent.ydata)))
            self.update_sorting_point(coords)

    def update_sorting_point(self, coords):
        for p in self.selected_sorting_points:
            p.remove()
        self.selected_sorting_points = []

        point_ori = self.ori_ax.scatter(coords[0], coords[1],
                                                 edgecolors="red",
                                                 facecolors="none",
                                                 linewidths=1, zorder=1)

        point_wh = self.wh_ax.scatter(coords[0], coords[1],
                                                 edgecolors="red",
                                                 facecolors="none",
                                                 linewidths=1, zorder=1)

        point_fact = self.fact_ax.scatter(coords[0], coords[1],
                                              edgecolors="red",
                                              facecolors="none",
                                              linewidths=1, zorder=1)

        self.selected_sorting_points.append(point_ori)
        self.selected_sorting_points.append(point_wh)
        self.selected_sorting_points.append(point_fact)

        self.ori_fig.canvas.draw()
        self.fact_fig.canvas.draw()

        self.terms = InteractiveFactorPlot.sort_matrices_by_element(self.terms, coords, reverse=True)
        self.on_slider_move(self.slider.val)

    @staticmethod
    def sort_matrices_by_element(l, position, reverse=False):
        c, r = position
        return sorted(l, key=lambda A: A[r, c], reverse=reverse)

