from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import rmsprop
from theano import config
config.floatX = "float32"
from os.path import isfile
import numpy
import random

class Network:
    # -------------------------------------------------------------------------------------------
    # network models
    gships = None
    gships_name = 'gships'
    # number of players
    players = 2
    growth_initial = 0.0
    # patterns saved based on player ID that have been reviewed and will be used for training
    # main info: [[in], [out]]
    patterns = {}
    # loss variable
    loss = []
    # -------------------------------------------------------------------------------------------
    
    def __init__(self):
        self.players = 2

        self.gships = Sequential([
            Dense(64, input_dim= 19),
            Activation('tanh'),
            #Dense(64),
            #Activation('tanh'),
            Dense(1),
            Activation('tanh'),
        ])
        rmsprop_ = rmsprop(lr=0.000025)
        self.gships.compile(loss='mse',
                        optimizer=rmsprop_,
                        metrics=['accuracy'])

        if (isfile(self.gships_name + 'Network.h5')):
            self.gships.load_weights(self.gships_name + 'Network.h5')

        self.init_local_vars()
    def save(self):
        '''
        save the weights to a file
        '''
        self.gships.save_weights(self.gships_name + 'Network.h5')                      
    def init_local_vars(self):
        for p in range(1, self.players + 1):
            self.patterns[p] = []
    def save_pattern(self, pid, pattern):
        self.patterns[pid].append(pattern)
        if len(self.patterns[pid]) > 3000: 
            # count = sum(x['reward'] > 0 for x in self.patterns[pid])
            del self.patterns[pid][0]

    def train(self, turn, pid, game_ratios, future_state, is_terminal = False):
        BATCH_SIZE = 32
        count = len(self.patterns[pid])
        if count == 0: return # if nothing saved, return

        self.patterns[pid][count - 1]['in_'] = future_state
        self.patterns[pid][count - 1]['terminal'] = is_terminal
        reward = game_ratios['growth_r'] - game_ratios['growth_er']  
        reward += game_ratios['growth_r'] - game_ratios['growth_desired'] 
        reward *= 0.07 # make reward only 7% 

        #if game_ratios['growth_r'] > game_ratios['growth_er']: reward = 0.02
        #elif game_ratios['growth_r'] < game_ratios['growth_er']: reward = -0.02  
        #if game_ratios['growth_r'] > game_ratios['growth_desired']: reward += 0.01
        #elif game_ratios['growth_r'] < game_ratios['growth_desired']: reward -= 0.01
        # if self.patterns[pid][count - 1]['out'][0][0] >= 0.5: reward *= -1
        reward = max(min(reward, 1.0), -1.0)
        self.patterns[pid][count - 1]['reward'] = reward

        # return if we cannot create a batch
        if count < BATCH_SIZE: return

        alpha = 0.01
        gamma = 0.95
        minibatch = random.sample(self.patterns[pid], BATCH_SIZE)        
        inputs = numpy.zeros((BATCH_SIZE, 19))
        targets = numpy.zeros((BATCH_SIZE, 1))
        for i in range(0, len(minibatch)):

            state_t = minibatch[i]['in']
            action_t = minibatch[i]['out']
            reward_t = minibatch[i]['reward']
            state_t1 = minibatch[i]['in_']
            terminal = minibatch[i]['terminal']
            # if terminated, only equals reward

            inputs[i] = state_t
            Q_sa = numpy.max([self.gships.predict(s)[0][0] for s in state_t1 if s is not None])

            # comput target value
            if terminal: targets[i] = reward_t
            else: targets[i] = reward_t + gamma * Q_sa
            targets[i] = max(min(targets[i], 1.0), -1.0) # keep it in bounds

        self.gships.train_on_batch(inputs, targets)
                
NNetwork = Network()