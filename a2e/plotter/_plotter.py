import os
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.dates as mdates
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dropout, Flatten, Reshape

style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretty.mplstyle')
plt.style.use(style_path)


def plot_model_layer_weights(model: Model):
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            if isinstance(layer, Conv1D):
                plot_conv_weights(layer.get_weights()[0], '../out/test/')
            else:
                plot_dense_weights(layer.get_weights()[0])


def plot_dense_weights(weights):
    fig = plt.figure()
    #fig.suptitle('layer: ' + str(layer.name))

    plt.matshow(weights)

    ax = plt.gca()
    ax.grid(False)


def plot_conv_weights(weights, save_path):
    figure = plt.figure()
    number_of_filters = weights.shape[2]

    for i in range(0, number_of_filters, 1):
        axes = figure.add_subplot(4, ceil(number_of_filters/4), i+1)
        filter = []

        for j in range(0, weights.shape[0], 1):
            filter.append(weights[j][0][i])

        axes.set_ylim([-0.5, 0.5])

        if i < 12:
            axes.set_xticklabels([])

        if i % 4 != 0:
            axes.set_yticklabels([])

        plt.xticks(np.arange(0, 4, 1))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.plot(filter)

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        figure.savefig(os.path.join(save_path, 'bla.pdf'), format='pdf', dpi=400)


def plot(x=None, y=None, ylabel=None, xlabel=None, label=None, out_path=None, show=False, show_legend=True, time_format=False, out_formats=['pdf', 'png'], title=None, xlim=None, ylim=None, color=None, show_screw_tightened=False, anomalous_start=None, create_figure=True):
    if create_figure:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        ax = plt.gca()

    if time_format:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(30))

    if x is None:
        x = range(0, len(y))

    ax.plot(x, y, color=color, label=label)

    if show_screw_tightened:
        ax.axvline(x=anomalous_start, color='#D4373E', linestyle='solid', label='screw_tightened')

    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=15)

    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=15)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_legend:
        ax.legend()

    if out_path:
        for out_format in out_formats:
            fig.savefig(out_path + '.' + out_format, format=out_format)

    if show:
        plt.show()
