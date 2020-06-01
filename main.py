import numpy as np
import torch
import gym
import argparse
import os
import datetime

from torch.utils.tensorboard import SummaryWriter

ENV_LEN = {
	'FrozenLake-v0': 100,
	'FrozenLake8x8-v0': 400,
}

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		# while not done:
		for _ in range(ENV_LEN[env_name]):
			action = policy.select_action(np.array(state), test=True)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
		
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="Q_UCB")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="FrozenLake-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--delay", default=1, type=int)       
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e7, type=int)   # Max time steps to run environment
	parser.add_argument("--gamma", default=0.99)                 # Discount factor
	parser.add_argument("--epsilon", default=0.1)             
	parser.add_argument("--delta", default=0.1)                

	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--comment", default="")
	parser.add_argument("--exp_name", default="exp_May_31")
	parser.add_argument("--which_cuda", default=0, type=int)

	args = parser.parse_args()

	device = torch.device('cuda:{}'.format(args.which_cuda))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	file_name = "{}_{}_{}_d_{}".format(args.policy, args.env, args.seed, args.delay)
	file_name += "_{}".format(args.comment) if args.comment != "" else ""
	folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name
	result_folder = 'runs/{}'.format(folder_name) 
	if args.exp_name is not "":
		result_folder = '{}/{}'.format(args.exp_name, folder_name)
	if args.debug: 
		result_folder = 'debug/{}'.format(folder_name)

	print("---------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
	print("---------------------------------------")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	kwargs = {
		"env": args.env,
		'num_state': env.observation_space.n,
		'num_action': env.action_space.n,
		"gamma": float(args.gamma),
		'epsilon': float(args.epsilon),
		'delta': float(args.delta),
		"device": device,
	}

	# Initialize policy
	Q_UCB = __import__(args.policy)
	policy = Q_UCB.Q_UCB(**kwargs)

	# replay_buffer = utils.ReplayBufferTorch(state_dim, action_dim, device=device)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	reward_queue = []

	writer = SummaryWriter(log_dir=result_folder, comment=file_name)

	#record all parameters value
	with open("{}/parameters.txt".format(result_folder), 'w') as file:
		for key, value in vars(args).items():
			file.write("{} = {}\n".format(key, value))

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		action = policy.select_action(np.array(state))

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		# writer.add_scalar('test/reward', reward, t+1)

		reward_queue.append(reward)

		# Train agent after collecting sufficient data
		if episode_timesteps <= args.delay:
			policy.train(state, action, None, next_state, writer=writer)
		else:
			policy.train(state, action, reward_queue.pop(0), next_state, writer=writer)

		state = next_state
		episode_reward += reward
		if episode_timesteps >= ENV_LEN[args.env]: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			# print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			reward_queue = []
			policy.reset_for_new_episode()

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluation = eval_policy(policy, args.env, args.seed)
			evaluations.append(evaluation)
			writer.add_scalar('test/avg_return', evaluation, t+1)
			np.save("{}/evaluations".format(result_folder), evaluations)
	for i in range(16):
		print(policy.Q[i])
		
