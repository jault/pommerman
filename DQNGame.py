#!/usr/bin/python
"""A simple Free-For-All game with Pommerman."""
import pommerman as pommerman
from pommerman import agents
from DQNAgent import DQNAgent
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from pommerman.agents import BaseAgent
from stable_baselines.deepq import PrioritizedReplayBuffer
import Constants as C
import random


class DullAgent(BaseAgent):

    def __init__(self):
        BaseAgent.__init__(self)
        self.simple = agents.SimpleAgent()

    def act(self, obs, action_space):
        step = int(obs['step_count'])
        if step > 100:
            return 5
        act = self.simple.act(obs, action_space)
        if act == 5:
            return random.choice(range(5))
        return act


def main():
    network = DuelingCNN(C.FILENAME, [5], (11, 11, 3), C.NUM_ACTIONS, C.Q_LEARNING_RATE)
    target = network.copy()
    replay = PrioritizedReplayBuffer(C.REPLAY_CAPACITY, alpha=C.ALPHA)

    agt = DQNAgent(network, replay)

    agent_list = [
        agt,
        DullAgent(),
        DullAgent(),
        DullAgent()
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    total_time = 0
    for i_episode in range(1000000):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            total_time += 1
            if total_time % C.UPDATE_RATE == 0:
                target = network.copy()
            if i_episode > 3: train(network, target, replay)
        print('Episode {} finished'.format(i_episode))
        if i_episode % 20 == 0:
            network.save()
    env.close()


def train(network, target, replay):
    states, actions, rewards, next_states, dones, weights, batch_idxes = replay.sample(C.BATCH_SIZE, beta=C.BETA)

    future_actions = network.best_actions(next_states)
    future_rewards = target.get_actions_values(next_states, future_actions)

    total_rewards = rewards + C.DISCOUNT * future_rewards
    dones = np.argwhere(dones > 0).flatten()
    for i in range(len(dones)):
        total_rewards[dones[i]] = rewards[dones[i]]

    predicted_rewards = network.get_actions_values(states, actions)
    errors = np.abs(predicted_rewards - total_rewards) + C.EPSILON

    network.train(states, actions, total_rewards, weights)

    replay.update_priorities(batch_idxes, errors)


keras.losses.huber_loss = tf.losses.huber_loss


def aggre(x):
    import tensorflow as tf
    state_value = x[0]
    action_advantage = x[1]
    return state_value + tf.subtract(action_advantage,
                     tf.reduce_mean(action_advantage, axis=-1, keep_dims=True))


class DuelingCNN(object):

    def __init__(self, filename, character_shape, field_shape, num_actions, learning_rate):
        self.filename = filename
        self.character_shape = character_shape
        self.field_shape = field_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        try:
            self.model = keras.models.load_model(filename)
            # self.update_learning_rate(learning_rate)    # should not be used with adaptive LR
        except (OSError, IOError):
            character = Input(shape=self.character_shape)
            field = Input(shape=self.field_shape)
            c1 = Conv2D(16, kernel_size=3, activation='relu')(field)
            c2 = Conv2D(32, kernel_size=2, activation='relu')(c1)
            c3 = Conv2D(64, kernel_size=2, activation='relu')(c2)

            flat_field = Flatten()(c3)

            merge = concatenate([character, flat_field])

            full = Dense(64, activation='relu')(merge)

            state_value_fc = Dense(64, activation='relu')(full)
            state_value = Dense(1)(state_value_fc)

            action_advantage_fc = Dense(64, activation='relu')(full)
            action_advantage = Dense(num_actions)(action_advantage_fc)

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            action_values = keras.layers.core.Lambda(aggre)([state_value, action_advantage])

            self.model = Model(inputs=[character, field], outputs=action_values)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='huber_loss')
            self.model.summary()

    def save(self):
        self.model.save(self.filename)

    def update_learning_rate(self, learning_rate):
        K.set_value(self.model.optimizer.lr, learning_rate)

    # Not a full copy, can not be used to train on
    def copy(self):
        cp_self = DuelingCNN(self.filename, self.character_shape, self.field_shape, self.num_actions, self.learning_rate)
        cp_model = keras.models.clone_model(self.model)
        cp_model.set_weights(self.model.get_weights())
        cp_self.model = cp_model
        return cp_self

    # Returns the best action given a single state
    def best_action(self, state):
        predicted = self.model.predict(self.unpack_states([state]))[0]
        action = np.argmax(predicted)
        value = predicted[action]
        return action, value

    def predict(self, state):
        predicted = self.model.predict(self.unpack_states([state]))[0]
        return predicted

    # Returns an array of best actions given an array of states
    def best_actions(self, states):
        predicted = self.model.predict_on_batch(self.unpack_states(states))
        return np.argmax(predicted, axis=1)

    # Returns an array of best action values given an array of states
    def best_value(self, states):
        predicted = self.model.predict_on_batch(self.unpack_states(states))
        return np.max(predicted, axis=1)

    # Returns the best action's value given a single state and action
    def get_action_value(self, state, action):
        return self.model.predict(self.unpack_states([state]))[0][action]

    # Returns an array of best action's values given an array of states and corresponding actions
    def get_actions_values(self, states, actions):
        predicted = self.model.predict_on_batch(self.unpack_states(states))
        values = np.empty(shape=len(actions))
        for i in range(len(actions)):
            values[i] = predicted[i][actions[i]]
        return values

    # Fits network to a batch of transitions while updating only the observed action
    def train(self, states, actions, utilities, weights):
        states = self.unpack_states(states)

        # It would be more efficient to modify huber_loss to not subtract from pred, but this is more concise
        predicted = self.model.predict_on_batch(states)
        for i in range(len(actions)):
            predicted[i][actions[i]] = utilities[i]

        if weights is not None:
            self.model.train_on_batch(states, predicted, weights)
        else:
            self.model.train_on_batch(states, predicted)

    def unpack_states(self, states):
        characters, fields = [], []
        for state in states:
            characters.append(state[0:5, -1, 0])
            fields.append(state[:, :-1, :])
        return [np.asarray(characters), np.asarray(fields)]


if __name__ == '__main__':
    main()
