"""This script will run the agent in a 1 D environment and plot its learning 
progress

Author: Johannes Cartus, 22.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import DiscreteEnvironment
from agent import AgentBase
from analysis import AnalyzerEpisode, AnalyzerRun, animate_episodes

import seaborn as sns

def generate_course():
    win = 30
    death = -30
    return np.array([
        [  0, 0, 0, 1, 0, win, 0, death, 1, 0, 0],
        [death, 0, 0, 4, 0,  0, 0,   0, 5, 0, 0],
        [death, 0, 0, 4, 0,  0, 0,   0, 5, 0, win]
    ]).transpose()

def main():

    game_map = generate_course()
    wins = np.arange(game_map.size)[game_map.flatten() == 30]
    deaths = np.arange(game_map.size)[game_map.flatten() == -30]

    environment = DiscreteEnvironment(
        game_map=game_map,
        init_mode="random",
        wins=wins,
        deaths=deaths
    )

    agent = AgentBase(
        environment=environment,
        discount_factor=0.7,
        learning_rate=0.1,
        epsilon_greedyness=0.2
    )

    n_episodes = 100
    agent.run(n_episodes=n_episodes)

    analysis = AnalyzerRun(run_statistics=agent._run_statistics)

    plt.figure(figsize=(6, 9))
    n = 3
    plt.subplot(n, 1, 1)
    analysis.plot_steps()
    plt.subplot(n, 1, 2)
    analysis.plot_reward()
    plt.subplot(n, 1, 3)
    analysis.plot_actions()

    plt.show()

    #TODO: plot whether agent won or died in episode vs episode
    
    animate_episodes(
        agent, 
        episodes=[0, n_episodes-3, n_episodes-2, n_episodes-1], 
        show=True, 
        save_path="animations",
        wins=wins,
        deaths=deaths
    )


    
        
if __name__ == '__main__':
    main()