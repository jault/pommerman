# Pommermand Deep Reinforcement Learning Agents

Project dependencies may be found in the requirements.txt file.

A pre-trained PPO agent is provided in the acritic.net and actor.net files. Executing the PPOGame.py file will bring up the game's GUI and load the pre-trained agent. To train a new agent remove the pre-trained agent files from the directory. The GUI has been modified to display the agent's decision making on the right hand side. For the DQNAgent these are the values assigned to actions (all zero indicating a uniformly random action), for the PPOAgent these are the probabilites of taking each action.

Please find a demo of the PPO agent uploaded to https://youtu.be/ZdzHKCza9j8