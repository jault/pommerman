from pommerman.agents import BaseAgent
from pommerman.agents.simple_agent import SimpleAgent
import numpy as np
import random
import Constants as C

INITIAL_STATE = np.zeros((11, 12, 3))


class DQNAgent(BaseAgent):

    def __init__(self, network, replay):
        BaseAgent.__init__(self)
        # Network and replay are defined outside of agent so that multiple
        # agents could be run at once sharing experience
        self.network = network
        self.replay = replay

        self.prev_state = INITIAL_STATE
        self.prev_action = 0
        self.prev_pos = np.ones((2,))
        self.step = 0
        self.simple = SimpleAgent()
        self.ep = 0

        self.acts = self.format_acts(np.ones((C.NUM_ACTIONS, )), 0)

    @staticmethod
    def format_acts(acts, act):
        acts = np.round(acts, 2)
        stringy = []
        for i in range(len(acts)):
            stringy.append("{: 4.2f}".format(acts[i]))
        stringy.append(str(act))
        return stringy

    def act(self, obs, action_space):
        state = self.process_obs(obs)
        if (self.prev_state == INITIAL_STATE).all():
            self.prev_state = state
        self.step = int(obs['step_count'])
        self.replay.add(self.prev_state, self.prev_action, 1, state, float(False))

        pos = np.array(obs['position'], dtype=np.float32)
        if (pos == self.prev_pos).all() and self.prev_action != 0 and self.prev_action != 5:
            self.replay.add(self.prev_state, 0, 1, state, float(False))
        self.prev_pos = pos

        predicted = np.zeros((C.NUM_ACTIONS,))
        if self.ep < 0:
            action = self.simple.act(obs, action_space)
        elif random.uniform(0, 100) < C.EXPLORE:
            action = random.choice(range(C.NUM_ACTIONS))
        else:
            predicted = self.network.predict(state)
            action = np.argmax(predicted)

        self.acts = self.format_acts(predicted, action)

        self.prev_state = state
        self.prev_action = action
        return action

    def episode_end(self, reward):
        self.ep += 1
        if reward < 1:
            reward = 0
            print(self.step)
        else:
            print(self.step, 'win!')
        self.replay.add(self.prev_state, self.prev_action, reward, INITIAL_STATE, float(True))
        self.prev_pos = np.ones((2,))
        self.prev_state = INITIAL_STATE
        self.prev_action = 0
        self.step = 0

    @staticmethod
    def process_obs(obs):
        position = np.array(obs['position'], dtype=np.float32)
        ammo = np.float32(obs['ammo'])
        blast_strength = np.float32(obs['blast_strength'])
        can_kick = np.float32(obs['can_kick'])
        ints = np.array([ammo, blast_strength, can_kick], dtype=np.float32)
        pad = np.zeros((6,), dtype=np.float32)
        character = np.concatenate((position, ints, pad))
        enemies = np.array([e.value for e in obs["enemies"]], dtype=np.float32)

        board = np.array(obs['board'], dtype=np.float32)

        for i in range(len(enemies)):
            board[board == enemies[i]] = -1

        board = np.c_[board, character]

        bomb_blast = np.array(obs['bomb_blast_strength'], dtype=np.float32)
        bomb_blast = np.c_[bomb_blast, np.zeros((11,), dtype=np.float32)]
        bomb_life = np.array(obs['bomb_life'], dtype=np.float32)
        bomb_life = np.c_[bomb_life, np.zeros((11,), dtype=np.float32)]

        field = np.stack((board, bomb_blast, bomb_life), axis=-1)

        return field
