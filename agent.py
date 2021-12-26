"""This module contians the implementation of the agent. 

Author: Johannes Cartus, 22.12.2021
"""

import numpy as np

from collections import defaultdict

from environment import EnvironmentBaseClass

class AgentBase(object):
    
    def __init__(
        self, 
        environment: EnvironmentBaseClass,
        discount_factor,
        learning_rate,
        epsilon_greedyness
    ):
        self._gamma = discount_factor
        self._alpha = learning_rate
        self._epsilon = epsilon_greedyness # probability for exploration
        
        self._env = environment

        self._q_table = np.zeros(shape=(self._env.n_states, self._env.n_actions))

    @property
    def _current_state(self):
        return self._env.current_state

    @property
    def _previous_state(self):
        return self._env.previous_state

    def _select_next_action(self):

        if np.random.rand() <= self._epsilon:

            # choose random action
            action = np.random.choice(self._env.actions)

        else:
            action = self._find_q_optimal_action(self._current_state)

        return action

    def _find_q_optimal_action(self, current_state):
        """Find the action which as the highest entry in the Q table for the 
        given state"""
        return np.argmax(self._q_table[current_state])

    def _update_q_table(
        self, 
        old_state, 
        action, 
        new_state,
        reward
    ):
        """This is the Q Learning part of the project. We use the Bellman
        equation to update the Q-table"""

        q_old = self._q_table[old_state, action]

        q_new = \
            q_old + \
            self._alpha * (
                reward + \
                self._gamma * np.max(self._q_table[new_state]) \
                - q_old
            )  

        self._q_table[old_state, action] = q_new 

    def _measure_reward_of_last_action(self):
        return self._env.reward(
            old_state=self._env.previous_state,
            new_state=self._env.current_state
        )
    
    def run_episode(self):
        """This runs a single episode, i.e. one game run."""

        step_count = -1
        self._init_episode_statistic()

        # set the initial state
        self._env.initialize()

        while not self._env.is_episode_finished():
            
            step_count += 1

            
            # choose an action 
            action = self._select_next_action()

            # perform an action 
            self._env.perform_action(action=action)

            # measure reward
            reward = self._measure_reward_of_last_action()

            # update Q-table
            self._update_q_table(
                old_state=self._env.previous_state,
                action=action,
                new_state=self._env.current_state,
                reward=reward
            )

            # do some logging for later analysis & plotting
            self._collect_episode_statistics(step_count, action, reward)
    

    def run(self, n_episodes):

        for i in range(n_episodes):

            self.run_episode()

            self._collect_run_statistics()

    def _init_episode_statistic(self):
        """Init buffer for logging"""
        self._episode_statistics = defaultdict(list)

    def _collect_episode_statistics(self, step, action, reward):
        """Do some logging for later analysis"""

        self._episode_statistics["step"].append(step)
        self._episode_statistics["old_state"].append(self._env.previous_state)
        self._episode_statistics["new_state"].append(self._current_state)
        self._episode_statistics["action"].append(action)
        self._episode_statistics["reward"].append(reward)
        self._episode_statistics["q_table"].append(self._q_table.copy())

    def _collect_run_statistics(self):
        """Do more logging, over many episodes."""

        if not hasattr(self, "_run_statistics"):
            self._n_episodes_run = -1
            self._run_statistics = defaultdict(list)

        n_steps = self._env._n_actions_performed

        self._n_episodes_run += 1

        self._run_statistics["episode"] += [self._n_episodes_run]*n_steps
        self._run_statistics["n_steps"] += [n_steps]*n_steps
        

        for x in ["old_state", "new_state", "action", "reward"]:
            self._run_statistics[x] += self._episode_statistics[x]

        self._run_statistics["finish"] += \
            [self._evaluate_finial_state_of_env()]*n_steps

    def _evaluate_finial_state_of_env(self):
        """Get the state in which the agent has finished the game. 
        I.e. win/death/iterations exceeded"""
        state = self._env._evaluate_current_state()

        if state == "unfinished":
            if self._env._n_actions_performed == self._env._max_actions:
                state = "max_iter_exceeded"
            else:
                state = "other"

        return state

    @property
    def n_steps_run(self):
        try:
            return self._run_statistics["n_steps"]
        except AttributeError:
            return [0]


