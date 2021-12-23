"""This module will create plots from the collected statistic of the run
Author: Johannes Cartus, 22.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt

class AnalyzerEpisode(object):

    def __init__(self, episode_statistics):

        self.data = episode_statistics

        self.index = np.arange(len(self.data["state"]))

    def _set_font_defaults(self):
        """Sets size of font in labels etc, returns a suitable fig size"""
        #todo set fontsizes etc.
        
        # figsize
        return (8, 4)

    def plot_reward(self):
        
        figsize = self._set_font_defaults(self)

        plt.figure(figsize=figsize)

        plt.plot(self.index, self.data["reward"])
        
        plt.xlabel("nr. of action / 1", fontsize=16)
        plt.ylabel("Reward / 1", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)