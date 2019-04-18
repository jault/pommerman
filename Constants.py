
NUM_ACTIONS = 6
FILENAME = 'pomme.net'
EXPLORE = 10
Q_LEARNING_RATE = 0.001
PG_LEARNING_RATE = 0.000005
DISCOUNT = 0.997
UPDATE_RATE = 500
BATCH_SIZE = 64
REPLAY_CAPACITY = 50000

ALPHA = 0.6
EPSILON = 0.00001
# Importance sampling - beta should reach 1.0 by convergence, increased by eta each transition recorded
BETA = 0.4
ETA = 0.00025