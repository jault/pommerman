# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

from keras.models import Model
import keras
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
from pommerman.agents import BaseAgent
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
import tensorflow as tf
import Constants as C


STATE_SHAPE = (11, 12, 3)


def ppo_loss(advantage, orig_prob):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * orig_prob
        weight = prob / (old_prob + 1e-10)
        clipped = K.clip(weight, min_value=1 - 0.2, max_value=1 + 0.2)
        mini = K.minimum(weight * advantage, clipped * advantage)
        entrop = 1e-3 * (prob * K.log(prob + 1e-10))
        return -K.mean(mini + entrop)
    return loss


keras.losses.ppo_loss = ppo_loss


class PPOAgent(BaseAgent):
    def __init__(self):
        BaseAgent.__init__(self)
        self.actor = self.actor_model()
        self.critic = self.critic_model()

        self.episode = 0
        self.state = np.zeros(shape=STATE_SHAPE)
        self.reward = []

        self.state_action_probs = [[], [], []]
        self.action_mask = np.zeros((C.NUM_ACTIONS,))
        self.action_probs = [np.zeros((C.NUM_ACTIONS,))]
        self.acts = self.format_acts(np.ones((C.NUM_ACTIONS,)), 0)
        self.prev_reward = 45

    @staticmethod
    def actor_model():
        field = Input(shape=STATE_SHAPE)
        advantage = Input(shape=(1,))
        orig_prob = Input(shape=(C.NUM_ACTIONS,))

        c1 = Conv2D(32, kernel_size=3, activation='elu')(field)
        c2 = Conv2D(64, kernel_size=2, activation='elu')(c1)
        c3 = Conv2D(128, kernel_size=2, activation='elu')(c2)

        flat_field = Flatten()(c3)
        fc1 = Dense(128, activation='elu')(flat_field)
        full = Dense(64, activation='elu')(fc1)

        out = Dense(C.NUM_ACTIONS, activation='softmax')(full)

        actor_model = Model(inputs=[field, advantage, orig_prob], outputs=out)
        actor_model.compile(optimizer=Adam(C.PG_LEARNING_RATE), loss=ppo_loss(advantage=advantage, orig_prob=orig_prob))

        try:
            actor_model.load_weights('actor.net')
        except (OSError, IOError):
            actor_model.summary()

        return actor_model

    @staticmethod
    def critic_model():
        try:
            critic_model = keras.models.load_model('acritic.net', custom_objects={'tf': tf})
        except (OSError, IOError):
            field = Input(shape=STATE_SHAPE)
            c1 = Conv2D(32, kernel_size=3, activation='elu')(field)
            c2 = Conv2D(64, kernel_size=2, activation='elu')(c1)
            c3 = Conv2D(128, kernel_size=2, activation='elu')(c2)

            flat_field = Flatten()(c3)
            fc1 = Dense(128, activation='elu')(flat_field)
            full = Dense(64, activation='elu')(fc1)

            out = Dense(1)(full)

            critic_model = Model(inputs=field, outputs=out)
            critic_model.compile(optimizer=Adam(C.PG_LEARNING_RATE), loss='mse')
            critic_model.summary()

        return critic_model

    def get_action(self):
        probs = self.actor.predict([[self.state], np.zeros((1, 1)), np.zeros((1, C.NUM_ACTIONS))])
        action = np.random.choice(C.NUM_ACTIONS, p=probs[0])
        action_mask = np.zeros(C.NUM_ACTIONS)
        action_mask[action] = 1
        return action, action_mask, probs

    def discount_future(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * C.DISCOUNT

    def collect_reward(self):
        board = self.state[:, :-1, 0].copy()
        board[board == 3] = 0
        board[board == 4] = 0
        num_passages = 121 - len(np.nonzero(board)[0])
        space_maker = (num_passages - self.prev_reward)
        if space_maker < 0: space_maker = 0
        self.reward.append(space_maker)
        self.prev_reward = num_passages

    def act(self, obs, action_space):
        self.state_action_probs[0].append(self.state)
        self.state_action_probs[1].append(self.action_mask)
        self.state_action_probs[2].append(self.action_probs)
        self.state = self.process_obs(obs)    # (11,12,3)
        self.collect_reward()

        action, self.action_mask, self.action_probs = self.get_action()

        self.acts = self.format_acts(self.action_probs[0], action)

        return action

    def episode_end(self, reward):
        print('steps', len(self.state_action_probs[0]), 'rew', self.prev_reward)
        self.discount_future()

        # Unpack episode experience
        batch = [[], [], [], []]
        for i in range(len(self.state_action_probs[0])):
            obs, action, pred = self.state_action_probs[0][i], self.state_action_probs[1][i], self.state_action_probs[2][i]
            r = self.reward[i]
            batch[0].append(obs)
            batch[1].append(action)
            batch[2].append(pred)
            batch[3].append(r)

        # Reset variables for next episode
        self.episode += 1
        self.state = np.zeros(shape=STATE_SHAPE)
        self.reward = []
        self.state_action_probs = [[], [], []]
        self.prev_reward = 45
        self.action_mask = np.zeros((C.NUM_ACTIONS,))
        self.action_probs = [np.zeros((C.NUM_ACTIONS,))]

        # Shape batch for models
        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(
            np.array(batch[3]), (len(batch[3]), 1))
        old_prediction = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        # Calculate advantage function
        pred_values = self.critic.predict(obs)
        advantage = reward - pred_values

        # Update models using this episodes experience
        self.actor.fit([obs, advantage, old_prediction], action, batch_size=C.BATCH_SIZE, shuffle=True, verbose=0)
        self.critic.fit(obs, reward, batch_size=C.BATCH_SIZE, shuffle=True, verbose=0)

        if self.episode % 20 == 0:
            self.actor.save_weights('actor.net')
            self.critic.save('acritic.net')

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

    @staticmethod
    def format_acts(acts, act):
        acts = np.round(acts, 2)
        stringy = []
        for i in range(len(acts)):
            stringy.append("{: 4.2f}".format(acts[i]))
        stringy.append(str(act))
        return stringy
