#!/usr/bin/python
import pommerman as pommerman
from pommerman import agents
from PPOAgent import PPOAgent
from pommerman.agents import BaseAgent
import random


class DullAgent(BaseAgent):

    def __init__(self):
        BaseAgent.__init__(self)
        self.simple = agents.SimpleAgent()

    def act(self, obs, action_space):
        act = self.simple.act(obs, action_space)
        if act == 5:
            return random.choice(range(5))
        return act


def main():
    agt = PPOAgent()

    agent_list = [
        agt,
        DullAgent(),
        DullAgent(),
        DullAgent()
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    total_time = 0
    for i_episode in range(1000000):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            total_time += 1
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
