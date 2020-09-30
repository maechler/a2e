import os
import matplotlib.pyplot as plt
from a2e.utility import to_absolute_path, out_path
import matplotlib.dates as mdates

style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretty.mplstyle')
plt.style.use(style_path)


def plot_model_layer_weights(model):
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            fig = plt.figure()
            fig.suptitle('layer: ' + str(layer.name))

            plt.matshow(layer.get_weights()[0])

            ax = plt.gca()
            ax.grid(False)

def plot(x=None, y=None, ylabel='', out_name=None, xlabel='time [h]', show=False, time_format=True, title=None, xlim=None, ylim=None, color=None, show_screw_tightened=False, anomalous_start=None):
    fig, ax = plt.subplots()

    if time_format:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(30))

    ax.plot(x, y, color=color)

    if show_screw_tightened:
        ax.axvline(x=anomalous_start, color='#D4373E', linestyle='solid', label='screw_tightened')

    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)

    #if title is not None:
    #    ax.set_title(self.get_plot_title(title))

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if out_name:
        fig.savefig(out_path(out_name + '.pdf'), format='pdf')
        fig.savefig(out_path(out_name + '.png'), format='png')

    if show:
        plt.show()
