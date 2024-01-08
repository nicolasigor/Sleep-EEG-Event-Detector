# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.interpolate import interp1d

from src.common import viz, constants


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
            self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black",
                 barwidth=None,
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, \
            TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(
                Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth,
                          fc="none"))
        if sizey:
            bars.add_artist(
                Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth,
                          fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx, minimumdescent=False)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0,
                           sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0,
                           sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False,
                                   **kwargs)


def add_scalebar(ax, matchx=False, matchy=False, hidex=True, hidey=True,
                 **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb


def set_axis_color(ax, color=viz.AXIS_COLOR):
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.title.set_color(color)
    return ax


def set_legend_color(lg, color=viz.AXIS_COLOR):
    plt.setp(lg.get_texts(), color=color)
    return lg


def piecewise_constant_histogram(bins, counts):
    n_bins = len(bins) - 1
    x_list = [bins[0]]
    y_list = [0]
    for i in range(n_bins):
        x_list.extend([bins[i], bins[i+1]])
        y_list.extend(2 * [counts[i]])
    x_list.append(bins[-1])
    y_list.append(0)
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    return x_list, y_list


def find_bin_idx(value, bins):
    n_bins = len(bins) - 1
    if value < bins[0] or value > bins[-1]:
        raise ValueError('Value outside bin boundaries')
    for i in range(n_bins):
        start_bin = bins[i]
        if value < start_bin:
            return i-1
    return n_bins


def format_metric_vs_iou_plot(
        ax,
        metric_name,
        iou_thr_to_show=None
):
    if iou_thr_to_show is not None:
        ax.axvline(
            iou_thr_to_show,
            linestyle='-',
            color=viz.GREY_COLORS[4],
            zorder=1,
            linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticks(np.linspace(0, 1, 11), minor=True)
    ax.set_yticks(np.linspace(0.2, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 11), minor=True)
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.set_xlabel('IoU Threshold', fontsize=viz.FONTSIZE_GENERAL)
    ax.set_ylabel(metric_name, fontsize=viz.FONTSIZE_GENERAL)
    ax.yaxis.grid(which='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    return ax


def format_duration_scatter_plot(ax, min_dur, max_dur, xlabel, ylabel):
    ax.set_xlim([min_dur, max_dur])
    ax.set_ylim([min_dur, max_dur])
    ax.plot(
        [min_dur, min_dur], [max_dur, max_dur],
        linestyle='-', color=viz.GREY_COLORS[4], zorder=10, linewidth=1)
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.set_xlabel(xlabel, fontsize=viz.FONTSIZE_GENERAL)
    ax.set_ylabel(ylabel, fontsize=viz.FONTSIZE_GENERAL)
    return ax


def format_iou_hist_plot(
        ax,
        label_pos_list,
        label_txt_list,
        iou_major_ticks=np.linspace(0, 1, 6),
        iou_minor_ticks=np.linspace(0, 1, 11)
):
    ax.set_xlim([0.0, 1.0])
    ax.set_xticks(iou_major_ticks)
    ax.set_xticks(iou_minor_ticks, minor=True)
    ax.set_yticks(label_pos_list)
    ax.set_yticklabels(label_txt_list)
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.set_xlabel('IoU of Matching', fontsize=viz.FONTSIZE_GENERAL)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    ax.yaxis.tick_right()
    return ax


def format_precision_recall_plot(
        ax,
        show_diagonal=True,
        show_iso_f1=False,
        axis_lims=(0, 1),
        pr_major_ticks=np.linspace(0, 1, 6),
        pr_minor_ticks=np.linspace(0, 1, 11),
        skip_first_ytick=True
):
    if show_iso_f1:
        ax = show_isof1(ax, axis_lims[0], axis_lims[1])

    if show_diagonal:
        ax.plot(
            axis_lims, axis_lims,
            linewidth=1, color=viz.GREY_COLORS[4],
            zorder=1)
    ax.set_xlabel('Recall', fontsize=viz.FONTSIZE_GENERAL)
    ax.set_ylabel('Precision', fontsize=viz.FONTSIZE_GENERAL)
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.set_xlim(axis_lims)
    ax.set_ylim(axis_lims)
    if skip_first_ytick:
        ax.set_yticks(pr_major_ticks[1:])
    else:
        ax.set_yticks(pr_major_ticks)
    ax.set_yticks(pr_minor_ticks, minor=True)
    ax.set_xticks(pr_major_ticks)
    ax.set_xticks(pr_minor_ticks, minor=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    return ax


def format_legend(ax, external_legend=True, remove_alpha=True):
    if external_legend:
        lg = ax.legend(
            loc='lower left', labelspacing=viz.LEGEND_LABEL_SPACING,
            fontsize=viz.FONTSIZE_GENERAL,
            frameon=False,
            bbox_to_anchor=(1, 0), ncol=1)
    else:
        lg = ax.legend(
            loc='lower left', labelspacing=viz.LEGEND_LABEL_SPACING,
            fontsize=viz.FONTSIZE_GENERAL,
            frameon=False)
    if remove_alpha:
        for lh in lg.legendHandles:
            lh.set_alpha(1.0)
            lh._legmarker.set_alpha(1.0)
    return lg


def densify_curve(x, y, factor=5):
    inter_fn = interp1d(x, y, kind='cubic')
    n_points = len(x) * factor
    x_denser = np.linspace(np.min(x), np.max(x), num=n_points, endpoint=True)
    y_denser = inter_fn(x_denser)
    return x_denser, y_denser


def average_curves(x_list, y_list):
    x_min = np.max([np.min(single_x) for single_x in x_list])
    x_max = np.min([np.max(single_x) for single_x in x_list])
    n_points = np.max([len(single_x) for single_x in x_list])
    x = np.linspace(x_min, x_max, num=n_points, endpoint=True)
    new_y_list = []
    for k in range(len(x_list)):
        single_x = x_list[k]
        single_y = y_list[k]
        fn_interp = interp1d(single_x, single_y, kind='linear')
        new_single_y = fn_interp(x)
        new_y_list.append(new_single_y)
    y = np.stack(new_y_list, axis=1).mean(axis=1)
    return x, y


def show_isof1(ax, min_level, max_level):
    delta = 0.01
    x_ = np.arange(1, 100) * delta
    y_ = np.arange(1, 100) * delta
    X, Y = np.meshgrid(x_, y_)
    Z = 2 * X * Y / (X + Y)

    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    levels = [level for level in levels if min_level < level < max_level]

    CS = ax.contour(X, Y, Z, colors='k', alpha=0.3, levels=levels)
    ax.clabel(CS, fontsize=7.5, fmt='%1.2f')
    return ax


def add_label_chart(ax, label, fontsize):
    ax.set_title(r"$\bf{%s}$"% label, fontsize=fontsize, loc='left')
    return ax


def format_precision_recall_plot_simple(
        ax,
        axis_markers=None,
        show_quadrants=True,
        show_grid=True,
        minor_axis_markers=None,
        axis_range=(0, 1)
):
    if axis_markers is None:
        axis_markers = np.arange(axis_range[0], axis_range[1] + 0.001, 0.1)
    if minor_axis_markers is None:
        minor_axis_markers = axis_markers
    # Diagonal
    ax.plot(axis_range, axis_range, zorder=1, linewidth=1, color=viz.GREY_COLORS[4])
    # Square basis
    ax.set_xlim([axis_range[0], axis_range[1]])
    ax.set_ylim([axis_range[0], axis_range[1]])
    ax.set_aspect('equal')
    # Ticks
    ax.set_yticks(axis_markers)
    ax.set_xticks(axis_markers)
    ax.set_xticks(minor_axis_markers, minor=True)
    ax.set_yticks(minor_axis_markers, minor=True)
    if show_grid:
        ax.grid(which="minor")
    if show_quadrants:
        ax.axhline(0.5, color=viz.GREY_COLORS[5], linewidth=2)
        ax.axvline(0.5, color=viz.GREY_COLORS[5], linewidth=2)


def get_fold_colors():
    color_dict = {
        0: viz.PALETTE[constants.RED],
        1: viz.PALETTE[constants.BLUE],
        2: viz.PALETTE[constants.GREEN],
        3: viz.PALETTE[constants.DARK],
        4: viz.PALETTE[constants.PURPLE],
        5: viz.PALETTE[constants.CYAN],
        6: viz.PALETTE[constants.GREY],
    }
    return color_dict


def get_performance_string(outputs):
    perf_str = "F1: %1.1f\u00B1%1.1f, mIoU: %1.1f\u00B1%1.1f\nR: %1.1f\u00B1%1.1f, P: %1.1f\u00B1%1.1f" % (
        100 * np.nanmean(outputs['f1']), 100 * np.nanstd(outputs['f1']),
        100 * np.nanmean(outputs['miou']), 100 * np.nanstd(outputs['miou']),
        100 * np.nanmean(outputs['rec']), 100 * np.nanstd(outputs['rec']),
        100 * np.nanmean(outputs['prec']), 100 * np.nanstd(outputs['prec']),
    )
    return perf_str
