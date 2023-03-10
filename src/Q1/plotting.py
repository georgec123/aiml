import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def parse_wld(wld: list, played_first: bool = True) -> Tuple[list, float]:
    """
    Parse the wld list to get the win percentage for a given player

    :param wld: list of tuples (result, played_first)
    :param played_first: player to get win percentage for  

    :return: list of results for given player, win percentage
    """
    turn_wld = [x[0] for x in wld if x[1] is played_first]
    win_pct = 100 * \
        np.sum([1 if x == 'w' else 0 for x in turn_wld]) / len(turn_wld)

    return turn_wld, win_pct


def plot_wld(wld: list) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the win/loss/draw results for both players

    
    :param wld: list of tuples (result, played_first)

    :return: figure, axes
    """
    def subplot(wld_arr, ax):

        ws = np.cumsum([1 if x == 'w' else 0 for x in wld_arr])
        ds = np.cumsum([1 if x == 'l' else 0 for x in wld_arr])
        ls = np.cumsum([1 if x == 'd' else 0 for x in wld_arr])

        ax.plot(ws, label='w')
        ax.plot(ds, label='d')
        ax.plot(ls, label='l')

        return ax

    first_wld, first_win_pct = parse_wld(wld, played_first=True)
    second_wld, second_win_pct = parse_wld(wld, played_first=False)

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0] = subplot(first_wld, axs[0])
    axs[1] = subplot(second_wld, axs[1])

    axs[0].set_title(f'Played First - {first_win_pct:.2f}%')
    axs[1].set_title(f'Played Second - {second_win_pct:.2f}%')

    axs[0].set_xlabel('Games')
    plt.legend()

    return fig, axs
