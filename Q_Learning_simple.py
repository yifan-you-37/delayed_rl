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

		# self.Q = defaultdict(lambda: np.zeros(self.num_action))
		self.Q = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)
		self.N = defaultdict(lambda: np.zeros(self.num_action))

		self.gamma = gamma
		self.delta = delta
		
	def select_action(self, state):
		return np.argmax(self.Q[int(state)])

	def alpha_k(self, k):
		# return 1. / k
		return 0.1
		
	def train(self, state, action, reward, next_state, replay_buffer=None, writer=None):		
		alpha = 0.1
		self.Q[int(state)][action] = (1 - alpha) * self.Q[int(state)][action] + alpha * (reward + self.gamma * np.max(self.Q[int(next_state)]))