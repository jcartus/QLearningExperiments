"""This script will run the agent in a 1 D environment and plot its learning 
progress

Author: Johannes Cartus, 22.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import DiscreteEnvironment1D
from agent import AgentBase
from analysis import AnalyzerEpisode, AnalyzerRun, animate_episodes

import seaborn as sns

def generate_course():
    return np.array([0, 0, 0, 1, 0, 16, 0, -20, 1, 0, 0])

def main():

    game_map = generate_course()

    environment = DiscreteEnvironment1D(game_map=game_map)

    agent = AgentBase(
        environment=environment,
        discount_factor=0.7,
        learning_rate=0.1,
        epsilon_greedyness=0.2
    )

    agent.run(n_episodes=50)

    analysis = AnalyzerRun(run_statistics=agent._run_statistics)

    plt.figure(figsize=(6, 9))
    n = 3
    plt.subplot(n, 1, 1)
    analysis.plot_steps()
    plt.subplot(n, 1, 2)
    analysis.plot_reward()
    plt.subplot(n, 1, 3)
    analysis.plot_actions()

    plt.figure()
    sns.heatmap(agent._q_table)
    
    animate_episodes(agent, episodes=[0, 49], show=True, save_path="animations")


    plt.show()
        
if __name__ == '__main__':
    main()