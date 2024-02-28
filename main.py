import numpy as np
import torch
import gym
import argparse
import os

import utils
import DPD3
import ESWD3


def softmax(x):
	sum_exp = 0
	for i in x:
		sum_exp += np.exp(i)
	result = np.exp(x) / sum_exp
	return result
def choose_algorithm(policy, env_name, seed, eval_episodes=100):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	real_value = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		n = 0
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			real_value += reward * (0.99 ** n)
			n += 1
	real_value /= eval_episodes

	algorithm_list = ['DDPG', 'TD3', 'REDQ', 'MCDDPG','TADD', 'QMD3']
	eval_list = np.array([])
	for i in range(len(algorithm_list)):
		state = eval_env.reset()
		action = policy.select_action(state)
		eval_q = policy.evalue(state, algorithm_list[i], action)
		eval_q = eval_q.cpu()
		eval_list = np.append(eval_list, eval_q.detach().numpy())

	eval_list = eval_list - real_value
	index = np.argmin(np.abs(eval_list - 0))

	print("---------------------------------------")
	print(f"Current true value: {real_value:.3f}")
	print('The deviation of each algorithm from the true value：')
	print(f"DDPG: {eval_list[0]:.3f} TD3: {eval_list[1]:.3f} REDQ: {eval_list[2]:.3f} MCDDPG: {eval_list[3]:.3f} TADD: {eval_list[4]:.3f} QMD3: {eval_list[5]:.3f}")
	print(f"The currently selected algorithm is：{algorithm_list[index]}")
	print("---------------------------------------")
	return algorithm_list[index]

def update_weight(policy, env_name, seed, weight_list, weight_var, eval_episodes=100):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	real_value = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		n = 0
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			real_value += reward * (0.99 ** n)
			n += 1
	real_value /= eval_episodes
	algorithm_list = ['DDPG', 'TD3', 'REDQ', 'MCDDPG','TADD', 'QMD3']
	eval_list = np.array([])
	for i in range(len(algorithm_list)):
		state = eval_env.reset()
		action = policy.select_action(state)
		eval_q = policy.evalue(state, algorithm_list[i], action)
		eval_q = eval_q.cpu()
		eval_list = np.append(eval_list, eval_q.detach().numpy())

	eval_list = eval_list - real_value
	eval_list = eval_list / real_value
	m = 0
	for i in eval_list:
		if i >= 0:
			weight_list[m] = -(i**2)/weight_var
		else:
			weight_list[m] = -(i**2)
		m += 1
	weight = softmax(weight_list)
	print("---------------------------------------")
	print(f"Current true value: {real_value:.3f}")
	print('The deviation of each algorithm from the true value：')
	print(f"DDPG: {eval_list[0]:.3f} TD3: {eval_list[1]:.3f} REDQ: {eval_list[2]:.3f} MCDDPG: {eval_list[3]:.3f} TADD: {eval_list[4]:.3f} QMD3: {eval_list[5]:.3f}")
	print('Weight of each algorithm：')
	print(f"DDPG: {weight[0]:.3f} TD3: {weight[1]:.3f} REDQ: {weight[2]:.3f} MCDDPG: {weight[3]:.3f} TADD: {weight[4]:.3f} QMD3: {weight[5]:.3f}")
	print("---------------------------------------")
	return weight

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="ESWD3")                # Policy name (DPD3 or ESWD3)
	parser.add_argument("--env", default="HalfCheetah-v3")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--choose_freq", default=1e4, type=int)     # How often (time steps) to choose an algorithm under DPD3 algorithm
	parser.add_argument("--weight_freq", default=1e4, type=int)     # How often (time steps) to update weights under ESWD3 algorithm
	parser.add_argument("--critic_number", default=5, type=int)     # The number of critics
	parser.add_argument("--weight_var", default=0.2, type=float)    # Weighting function's variance
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "DPD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["critic_number"] = args.critic_number
		policy = DPD3.DPD3(**kwargs)
	elif args.policy == "ESWD3":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["critic_number"] = args.critic_number
		policy = ESWD3.ESWD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	algorithm = 'TD3'
	weight_list = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166]
	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if args.policy == "DPD3":
				if t % args.choose_freq == 0 and t > args.choose_freq:
					algorithm = choose_algorithm (policy, args.env, args.seed)
				policy.train(replay_buffer, algorithm, args.batch_size)
			elif args.policy == "ESWD3":
				if t % args.weight_freq == 0 and t > args.weight_freq:
					weight_list = update_weight(policy, args.env, args.seed, weight_list, args.weight_var)
				policy.train(replay_buffer, weight_list, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
