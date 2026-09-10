# Create fig.add_ure and polar axes
from datetime import timedelta

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button


class InteractiveSeasonExploreButtons:
    def __init__(self, fig: Figure):

        ##################### Create button axes
        button_width = 0.15
        button_height = 0.1
        button_y1 = 0.4
        button_y2 = 0.2
        button_y3 = 0.6
        button_y4 = 0.8
        button_x1 = 0.2

        ax_d_minus = fig.add_axes([button_x1 + button_width, button_y1, button_width, button_height])
        ax_d_plus = fig.add_axes([button_x1 + 2 * button_width, button_y1, button_width, button_height])
        ax_m_minus = fig.add_axes([button_x1, button_y1, button_width, button_height])
        ax_m_plus = fig.add_axes([button_x1 + 3 * button_width, button_y1, button_width, button_height])
        ax_next_good_viirs = fig.add_axes([button_x1 + 2 * button_width, button_y2, button_width, button_height])
        ax_prev_good_viirs = fig.add_axes([button_x1 + button_width, button_y2, button_width, button_height])
        ax_next_good_s2 = fig.add_axes([button_x1 + 3 * button_width, button_y2, button_width, button_height])
        ax_prev_good_s2 = fig.add_axes([button_x1, button_y2, button_width, button_height])

        ax_a_minus = fig.add_axes([button_x1 + button_width, button_y3, button_width, button_height])
        ax_a_plus = fig.add_axes([button_x1 + 2 * button_width, button_y3, button_width, button_height])
        ax_mb_minus = fig.add_axes([button_x1 + button_width, button_y4, button_width, button_height])
        ax_mb_plus = fig.add_axes([button_x1 + 2 * button_width, button_y4, button_width, button_height])

        self.btn_d_minus = Button(ax_d_minus, "D-")
        self.btn_d_plus = Button(ax_d_plus, "D+")
        self.btn_m_minus = Button(ax_m_minus, "M-")
        self.btn_m_plus = Button(ax_m_plus, "M+")
        self.btn_next_good_viirs = Button(ax_next_good_viirs, "Next good VIIRS")
        self.btn_prev_good_viirs = Button(ax_prev_good_viirs, "Prev Good VIIRS")
        self.btn_next_good_s2 = Button(ax_next_good_s2, "Next good S2")
        self.btn_prev_good_s2 = Button(ax_prev_good_s2, "Prev Good S2")
        self.btn_a_minus = Button(ax_a_minus, "op. ob. param.")
        self.btn_a_plus = Button(ax_a_plus, "op. obs. param.+")
        self.btn_mb_minus = Button(ax_mb_minus, "member-")
        self.btn_mb_plus = Button(ax_mb_plus, "member+")
