from os import stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import configparser
import gym



class Agent:

    def __init__(self, nA=6, cfg_file='cfg.ini'):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.curr_epsiode = 1

        config = configparser.ConfigParser()
        config.read(cfg_file)

        self.algorithm = str(config['algorithm']['name'])
        self.gamma = float(config[self.algorithm]['gamma'])
        self.alpha = float(config[self.algorithm]['alpha'])
        self.eps = float(config[self.algorithm]['eps'])

        if self.algorithm == 'sarsa':
            self.update_Q = self.learn_Q_sarsa
        elif self.algorithm == 'sarsamax':
            self.update_Q = self.learn_Q_sarsamax
        elif self.algorithm == 'expected_sarsa':
            self.update_Q = self.learn_Q_sexpected_sarsa
        else:
            raise NotImplementedError

        


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)

        if np.random.random() > self.eps:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.randint(self.nA)
        return action  

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #### Select next action
        if self.algorithm == 'sarsa':
            next_action = self.select_action(next_state)
        else:
            next_action = None
        
        #### update Q
        self.Q[state][action] = self.update_Q(self.Q, state, action, next_state, next_action, reward, done, self.gamma, self.alpha, self.eps)        

        if done:
            # update epsiode count
            self.curr_epsiode += 1

            # update epsilon
            self.eps = self.get_epsilon(self.curr_epsiode)

    def get_epsilon(self, n_epsiode):
        """
        This function will change epsilon in a zigzag upto 1000 epsiodes.
        """

        if n_epsiode <= 1000:
            eps = 1 / (((n_epsiode-1) % 100) + 1)
        else:
            eps = 1 / (n_epsiode - 1000)
        return eps

    def learn_Q_sarsa(self, Q, state, action, next_state, next_action, reward, done, gamma, alpha, eps):
        predict = Q[state][action]
        target = reward + gamma *(0 if done else Q[next_state][next_action] )
        new = predict + alpha * (target - predict)

        return new

    def learn_Q_sarsamax(self, Q, state, action, next_state, next_action, reward, done, gamma, alpha, eps):
        predict = Q[state][action]
        target = reward + gamma *(0 if done else np.max(Q[next_state][:]))
        new = predict + alpha * (target - predict)
        return new

    def get_expected_Q(self, Q, state, eps):
        nA = len(Q[state][:])
        probs = np.zeros(nA)
        probs += eps/nA
        
        ## For most probable action 1 - eps would be added
        max_action = np.argmax(Q[state][:])
        probs[max_action] += (1-eps)
        return np.sum(Q[state][:] * probs)

    def learn_Q_sexpected_sarsa(self, Q, state, action, next_state, next_action, reward, done, gamma, alpha, eps):
        predict = Q[state][action]
        target = reward + gamma *(0 if done else self.get_expected_Q(Q, next_state, eps))
        new = predict + alpha * (target - predict)
        return new

