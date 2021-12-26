"""This module will create plots from the collected statistic of the run
Author: Johannes Cartus, 22.12.2021
"""

import os
from os.path import join

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

    def plot_average_final_state(self):
        x = self.df["episode"].unique()
        
        mean_rate = {f: [
            np.mean(self.df[
                (self.df.episode <= xi) & (self.df.step == 0)
            ]["finish"] == f) * 1e2 \
                for xi in x
        ] for f in ["win", "death", "max_iter_exceeded", "other"]}

        for f, y in mean_rate.items():
            plt.plot(x, y, "x-", label=f)
        
        
        plt.xlabel("episode / 1")
        plt.ylabel("rate of finishes / %")
        plt.legend()


class EpisodeAnimator(object):
    """Creates a nice animation of the moves done during an episode"""

    def __init__(self, game_map, states, wins=None, deaths=None):

        if len(game_map.shape) == 1:
            self.game_map = game_map.reshape(-1, 1)

        else:
            self.game_map = game_map
        

        self.states = states

        self.win = wins if not wins is None  else [np.argmax(self.game_map)]
        self.death = deaths if not deaths is None else [np.argmin(self.game_map)]

    def position_from_state(self, state):
        """State is a scalar index, get position on map"""
        try:
            x, y = np.unravel_index(state, self.game_map.shape)
        except ValueError:
            raise IndexError(f"State {state} out side of map (max: {self.game_map.size})")

        return x, y

    def annotate_field(self, state, txt):
        """Add text to a field on the map"""
        
        x, y = self.position_from_state(state)
        
        plt.text(x, y, txt, ha="center", va="center", color="gray", fontdict={"fontweight": "bold", "fontsize": 14})
        

    def mark_position(self, state, color="blue"):
        """Highlight a field on the map, e.g. to indicate 
        the current position"""
        from matplotlib.patches import Rectangle
        x, y = self.position_from_state(state)
        
        plt.gca().add_patch(Rectangle((x-0.5, y-0.5), 1, 1, fill=False, edgecolor=color, lw=3))


    def visualize_map(self):
    
        plt.imshow(self.game_map.transpose(), origin="lower")
        
        
        xticks = np.arange(self.game_map.shape[0])
        yticks = np.arange(self.game_map.shape[1])
        plt.xticks(xticks, xticks)
        plt.yticks(yticks, yticks)
        
        plt.xlabel("x")
        plt.ylabel("y")
        
        #cbar = plt.colorbar()
        #cbar.set_label("Gain")

    def visualize_map_and_landmarks(self, start_state):
    
        # plot the map
        self.visualize_map()

        # add wins
        for w in self.win:
            self.annotate_field(w, "Win")
        
        # add deaths
        for d in self.death:
            self.annotate_field(d, "Death")
        
        # mark the start position
        self.mark_position(start_state, color="grey")

    def make_animation(self, fig, interval=200, repeat=True):
        """Create and return an animation"""
        import matplotlib.animation as animation

        def init():
            self.visualize_map_and_landmarks(start_state=self.states[0])
            self.mark_position(self.states[0])
            plt.title("Step 0")

        def animate(i):
            plt.clf()
            self.visualize_map_and_landmarks(start_state=self.states[0])
            self.mark_position(self.states[i])
            plt.title(f"Step {i}")


        anim = animation.FuncAnimation(
            fig, 
            animate, 
            init_func=init, 
            frames=len(self.states), 
            repeat=repeat,
            interval=interval
        )
        
        return anim


def animate_episodes(agent, episodes=-1, save_path=None, show=False, wins=None, deaths=None):
    """Create an animation for moves in specified episodes."""

    game_map = agent._env.game_map

    df = pandas.DataFrame(agent._run_statistics)

    if episodes == -1:
        episodes = df["episode"].sort_values().unique()

    for episode in np.atleast_1d(episodes):
        
        dfp = df[df.episode==episode]

        # first old state must have been initial state
        states = [dfp.old_state.to_list()[0]] + dfp.new_state.to_list() 

        animator = EpisodeAnimator(
            game_map=game_map,
            states=states, 
            wins=wins,
            deaths=deaths
        )

        fig = plt.figure(figsize=(10, 4))

        anim = animator.make_animation(fig=fig)

        if save_path:
            anim.save(join(save_path, f"episode_{episode}.gif"))

        if show:
            plt.show()
