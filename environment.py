"""This module contains implementations of the environments.

Author: Johannes Cartus, 22.12.2021
"""

import numpy as np


class EnvironmentBaseClass(object):
    
    def __init__(self):

        self.previous_state = None
        
        self._actions = None
        self._states = None # the possible states to go to.

        self._n_actions_performed = 0

    @property
    def current_state(self):
        raise NotImplementedError("Depends on concrete implementation")

    @property
    def n_actions(self):
        try:
            return len(self._actions)
        except TypeError:
            raise NotImplementedError("Actions must be implemented in concrete envs")

    @property
    def n_states(self):
        try:
            return len(self._states)
        except TypeError:
            raise NotImplementedError("States depend on the map, to be impl. in concrete envs")
    
    @property
    def actions(self):
        return np.arange(len(self._actions))

    def initialize(self):
        """Set the inital state, etc."""

        self._n_actions_performed = 0

        raise NotImplementedError("Depends on the concrete states")

    def perform_action(self, action):
        """Performs the action / moves the player / updates the current state.
        
         - Should advance self._n_actions_performed += 1
         - MUST perform boundary checks and enforce periodicity
        """
        
        raise NotImplementedError("This will depend on the avialable actions.")

    def reward(self, old_state, new_state):
        raise NotImplementedError("Reward will depend on the concrete env.")
    
    def is_episode_finished(self):
        raise NotImplementedError("Criteria will depend on concrete env.")



class DiscreteEnvironment(EnvironmentBaseClass):

    def __init__(self, 
        game_map, 
        wins=None, 
        deaths=None, 
        max_actions=50,
        init_mode="zero"
    ):
        """Args:

        game_map <np.ndarray>: 1D array specifying the reward for moving to the 
            discrete positions. E.g. positive value is food, negative values an
            obstacle.
        """

        super(DiscreteEnvironment, self).__init__()

        self.game_map = game_map

        self.win_states = wins if wins else np.argmax(game_map)
        self.death_states = deaths if deaths else np.argmin(game_map)

        self._actions = self._generate_actions(self.game_map)

        self._max_actions = max_actions

        self._init_mode = init_mode

    def initialize(self):
        """Set the inital state"""

        if self._init_mode == "zero":
            # start at [0,0,..]
            self.current_position = np.zeros_like(self.game_map.shape)
        elif self._init_mode == "random":
            
            possible_states = np.arange(self.game_map.size)

            # remove wins and deaths
            possible_states = possible_states[~np.logical_or(
                np.isin(possible_states, self.win_states),
                np.isin(possible_states, self.death_states)
            )]

            self.current_position = np.array(np.unravel_index(
                np.random.choice(possible_states),
                self.game_map.shape
            ))

        else:
            raise ValueError("Unknown init mode: " + str(self._init_mode))

        self._n_actions_performed = 0

    @property
    def n_states(self):
        return self.game_map.size

    @staticmethod
    def _generate_actions(game_map):
        # Example 1D: +1 is move right, -1 is move left 
        actions = []
        for dim in range(len(game_map.shape)):
            if game_map.shape[dim] > 1:
                
                
                for direction in [+1, -1]:

                    action = np.zeros_like(game_map.shape)
                    action[dim] = direction
                    actions.append(action)
            
        return actions            

    def _state_from_position(self, position):
        """Calculates which state the current position is represented by"""

        if np.any(position < 0) or np.any(position > self.game_map.shape):
            raise ValueError("Invalid position: " + str(position)) 

        return np.ravel_multi_index(position, self.game_map.shape)

    @property
    def current_state(self):
        return self._state_from_position(self.current_position)

    def perform_action(self, action):
        
        # advance action counter
        self._n_actions_performed += 1

        # before action
        self.previous_state = self._state_from_position(self.current_position) 

        # perform the action
        self.current_position += self._actions[action]

        # boundary checks & enforcing periodicity
        self.current_position %= self.game_map.shape

    def reward(self, old_state, new_state):

        gain = self.game_map[new_state] - self.game_map[old_state] # state is just linear index

        if gain == 0:
            # if nothing was gained, punish the agent slightly
            return -1
        else:
            return gain

    def is_episode_finished(self):
        # after a few iters we kill the game
        max_iter_exceeded = self._n_actions_performed >= self._max_actions 
        
        # the worst field in the map is like dying
        has_died = np.isin(self.current_state, self.death_states)  # argmin gives linear index aka state ;)

        # the best field marks the goal
        has_reached_goal = np.isin(self.current_state, self.win_states)
        
        return max_iter_exceeded or has_died or has_reached_goal

class DiscreteEnvironment1D(DiscreteEnvironment):

    def __init__(self, game_map, *args, **kwargs):

        if len(game_map.shape) > 1:
            raise ValueError("Game map must be 1D, but is: " + str(game_map.shape))

        super(DiscreteEnvironment1D, self).__init__(
            game_map=game_map, 
            *args, 
            **kwargs
        )
