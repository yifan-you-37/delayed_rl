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
		epsilon=0.1,
		delta=0.1,
		device='cuda:0'
	):
		self.num_state = num_state
		self.num_action = num_action

		self.Q = 1/(1-gamma) * np.ones(num_state * 4).reshape(num_state, 4).astype(np.float32)
		self.Q_hat = 1/(1-gamma) * np.ones(num_state * 4).reshape(num_state, 4).astype(np.float32)
		self.N = np.zeros(num_state * 4).reshape(num_state, 4)

		self.R = np.ceil(np.log(3/(epsilon * (1-gamma))) / (1-gamma))
		self.M = np.log(1/((1-gamma) * epsilon))
		
		self.epsilon1 = epsilon / (24 * self.R * self.M * np.log(1/(1-gamma)))
		self.H = np.log(1/((1-gamma) * self.epsilon1)) / np.log(1/gamma)
		self.c2 = 4 * np.sqrt(2)
		
		self.gamma = gamma
		self.delta = delta
		self.total_it = 0
		self.log_freq = 10000

		self.queue = []
	def select_action(self, state, test=False):
		best_action = np.argmax(self.Q_hat[int(state)])
		return best_action

	def alpha_k(self, k):
		return 1. * (self.H + 1)/ (self.H + k)

	def small_L(self, k):
		return np.log(self.num_state * self.num_action * (k+1) * (k+2) / self.delta)
	
	def reset_for_new_episode(self):
		self.queue = []

	def train(self, state, action, reward, next_state, replay_buffer=None, writer=None):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)
			
		self.N[int(state)][action] = self.N[int(state)][action] + 1
		k = self.N[int(state)][action]
		
		# b = self.c2 / (1 - self.gamma) * np.sqrt(self.H * self.small_L(k) / k)
		
		const_mult = 30
		max_reward = 0.8
		# this constant needs to be tuned manually for each environment, for each reward delay.
		"""
		Constants used:
		FrozenLake-v0, delay 0: 1
		FrozenLake-v0, delay 1: 0.02
		FrozenLake-v0, delay 2: 0.50

		FrozenLake8x8-v0, delay 0: 1
		"""

		b = const_mult * np.sqrt(1/k) 

		next_V_hat = np.max(self.Q_hat[int(next_state)])

		alpha = self.alpha_k(k)
		self.Q[int(state)][action] = (1 - alpha) * self.Q[int(state)][action] + alpha * (max_reward + b + self.gamma * next_V_hat)
		self.queue.append((state, action, k))

		if reward is not None:
			old_state, old_action, old_k = self.queue.pop(0)
			old_alpha = self.alpha_k(old_k)
			self.Q[int(old_state)][old_action] = self.Q[int(old_state)][old_action] - old_alpha * (max_reward - reward)

		self.Q_hat[int(state)][action] = min(self.Q_hat[int(state)][action], self.Q[int(state)][action])
		if log_it:
			for i in range(self.Q.shape[0]):
				for j in range(self.Q.shape[1]):
					writer.add_scalar('train/Q_val_{}_{}'.format(i, j), self.Q[i][j], self.total_it)
			print(alpha, self.Q[int(state)])
			print(len(self.queue))