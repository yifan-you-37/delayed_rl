import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Q_UCB(object):
	def __init__(
		self,
		env,
		num_state,
		num_action,
		gamma=0.99,
		epsilon=0.01,
		delta=0.1,
		device='cuda:0'
	):
		self.num_state = num_state
		self.num_action = num_action

		self.Q = np.zeros(num_state * 4).reshape(num_state, 4).astype(np.float32)
		self.N = defaultdict(lambda: np.zeros(self.num_action))

		self.gamma = gamma
		self.delta = delta
		self.total_it = 0
		self.log_freq = 10000
	def select_action(self, state, test=False):
		if test is False:
			epsilon = 0.1
			action_probabilities = np.ones(self.num_action, dtype = float) * epsilon / self.num_action  
			best_action = np.argmax(self.Q[int(state)]) 
			action_probabilities[best_action] += (1.0 - epsilon) 
	
			action = np.random.choice(np.arange( 
				len(action_probabilities)), 
				p = action_probabilities) 
			return action
		else:
			return np.argmax(self.Q[int(state)])

	def alpha_k(self, k):
		# return 1. / k
		return 0.1

	def reset_for_new_episode(self):
		return

	def train(self, state, action, reward, next_state, replay_buffer=None, writer=None):
		if reward is None:
			return
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)
		alpha = 0.1
		self.Q[int(state)][action] = (1 - alpha) * self.Q[int(state)][action] + alpha * (reward + self.gamma * np.max(self.Q[int(next_state)]))

		if log_it:
			for i in range(self.Q.shape[0]):
				for j in range(self.Q.shape[1]):
					writer.add_scalar('train/Q_val_{}_{}'.format(i, j), self.Q[i][j], self.total_it)