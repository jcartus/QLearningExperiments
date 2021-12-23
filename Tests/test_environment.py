"""This module tests the funcitonality of environment 

Author: Johanes Cartus, 22.12.2021
"""

import pytest
import numpy as np

from environment import EnvironmentBaseClass, DiscreteEnvironment, DiscreteEnvironment1D

class TestDiscreteEnvironment1D(object):

    def test_enforces_1D(self):
        game_map = np.ones(shape=(10, 2))
        with pytest.raises(ValueError):
            env = DiscreteEnvironment1D(game_map=game_map)

class TestGenerationOfActions(object):

    def test_generation_from_1D_map(self):
        game_map = np.zeros(shape=10)

        env = DiscreteEnvironment(game_map=game_map)

        assert env.n_actions == 2 * len(game_map.shape)

        assert set(map(int, env.actions)) == set([1, -1])