# QLearningExperiments
This repo contains little experiments with reinforcement learning.

# The game 
To be exact I created a little minigame, 
where the player needs to move on an n-dimensional grid from the start position to the goal
While avoiding a few deadly obstacles. Each point of the grid may be assigned a 
value specifying how beneficial it is for the player to visit the point. 
For simplicity I used periodic boundary conditions.

Instead of struggling myself I set up an agent, 
which learns to master this mini-game via basic Q-learning.

# Agent performance

This is the first episode (untrained agent):
![Alt Text](https://github.com/jcartus/QLearningExperiments/blob/main/animations/before_training.gif?raw=true)


A few episodes later ...
![Alt Text](https://github.com/jcartus/QLearningExperiments/blob/main/animations/after_training.gif?raw=true)
