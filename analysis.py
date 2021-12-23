"""This module will create plots from the collected statistic of the run
Author: Johannes Cartus, 22.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas

class AnalyzerEpisode(object):

    def __init__(self, episode_statistics):

        self.data = episode_statistics

    def _set_font_defaults(self):
        """Sets size of font in labels etc, returns a suitable fig size"""
        #todo set fontsizes etc.
        
        # figsize
        return (8, 4)

    def plot_reward(self):
        
        figsize = self._set_font_defaults()

        plt.figure(figsize=figsize)

        plt.plot(self.data["steps"], self.data["reward"], "x-")
        
        plt.xlabel("nr. of action / 1", fontsize=16)
        plt.ylabel("Reward / 1", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    def animate_q_table(self, seconds_per_frame=2):
        
        self._set_font_defaults()
        fig, ax = plt.subplots()

        for i in self.data["steps"]:
            
            # plot Q Table 
            im = ax.imshow(self.data["q_table"][i])
            ax.set_title(f"Step nr: {i}")
            

            # add colorbar
            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes([0.7, 0.1, 0.04, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            # plot action
            #plt.text(self.data["old_state"][i], self.data["action"][i]+0.5, "sel.", fontdict={"fontsize": 20}, ha="center", va="center", color="w")
            #plt.text(self.data["new_state"][i], self.data["action"][i]+0.5, "sel.", fontdict={"fontsize": 20}, ha="center", va="center", color="w")

            # add labels            
            ax.set_xlabel("State / 1", fontsize=16)
            ax.set_ylabel("Action / 1", fontsize=16)
            #plt.legend()
            
            # Note that using time.sleep does *not* work here!
            plt.pause(seconds_per_frame)

            ax.cla()
            cbar_ax.cla()


class AnalyzerRun(object):

    def __init__(self, run_statistics):
        
        self.df = pandas.DataFrame(run_statistics)

    def plot_steps(self):
        
        dfp = self.df[["episode", "n_steps"]].drop_duplicates()

        x = dfp["episode"]
        y = dfp["n_steps"]

        plt.plot(x, y, "x-", label="steps")
        
        plt.xlabel("episode / 1")
        plt.ylabel("n_steps / 1")

    def plot_reward(self):

        x = self.df["episode"].unique()
        
        reward_mean = [np.mean(self.df[self.df.episode == xi])["reward"] for xi in x]
        reward_std = [np.std(self.df[self.df.episode == xi])["reward"] for xi in x]

        print(x, reward_mean)

        #plt.plot(x, reward_mean, "x-", label="reward")
        plt.errorbar(x, reward_mean, fmt="x-", yerr=reward_std, label="reward")

        plt.xlabel("episode / 1")
        plt.ylabel("reward / 1")

    def plot_actions(self):

        x = self.df["episode"].unique()
        
        for action in self.df.action.unique():
            y = [np.sum(self.df[self.df.episode == xi]["action"] == action) for xi in x]

            plt.plot(x, y, "x-", label=f"action {action}")

        plt.xlabel("episode / 1")
        plt.ylabel("x times selected / 1")
        plt.legend()