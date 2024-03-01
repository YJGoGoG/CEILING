import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1

	def ev(self, state, action):
		sa = torch.cat([state, action])

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



class DPD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		critic_number,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):
		self.critic_number = critic_number
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = []
		self.critic_target = []
		self.critic_optimizer = []

		for i in range(self.critic_number):
			self.critic.append(Critic(state_dim,action_dim).to(device))
			self.critic_target.append(copy.deepcopy(self.critic[i]))
			self.critic_optimizer.append(torch.optim.Adam(self.critic[i].parameters(), lr=3e-4))

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	def evalue(self, state, algorithm, action):
		state = torch.tensor(state).to(device).float()
		action = torch.tensor(action).to(device).float()
		Q_list=[]
		for i in range(self.critic_number):
			q=self.critic[i].ev(state, action)
			Q_list.append(q.unsqueeze_(0))
		Q_list = torch.cat(Q_list, dim=0)

		if algorithm == 'DDPG':
			eval_Q = Q_list[0]
		elif algorithm == 'TD3':
			eval_Q = torch.min(Q_list[0], Q_list[1])
		elif algorithm == 'REDQ':
			sub = random.sample(range(self.critic_number), 2)
			eval_Q = torch.min(Q_list[sub[0]], Q_list[sub[1]])
		elif algorithm == 'MCDDPG':
			eval_Q = Q_list.mean(axis=0, dtype=torch.float32)
		elif algorithm == 'TADD':
			eval_Q = 0.95 * torch.min(Q_list[0], Q_list[1]) + 0.05 * (
						Q_list[2] + Q_list[3]) / 2
		elif algorithm == 'QMD3':
			Q_list, indices = torch.sort(Q_list, dim=0)
			eval_Q = Q_list[self.critic_number // 2]
		return eval_Q
	def train(self, replay_buffer,algorithm, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
			target_Q_list=[]
			# Compute the target Q value
			for i in range(self.critic_number):
				target_temp_Q = self. critic_target[i](next_state,next_action)
				target_Q_list.append(target_temp_Q.unsqueeze_(0))
			target_Q_list =torch.cat(target_Q_list,dim=0)

			if algorithm == 'DDPG':
				target_Q = target_Q_list[0]
			elif algorithm == 'TD3':
				target_Q = torch.min(target_Q_list[0],target_Q_list[1])
			elif algorithm == 'REDQ':
				sub = random.sample( range(self.critic_number), 2 )
				target_Q = torch.min(target_Q_list[sub[0]],target_Q_list[sub[1]])
			elif algorithm == 'MCDDPG':
				target_Q = target_Q_list.mean(axis=0, dtype=torch.float32)
			elif algorithm == 'TADD':
				target_Q = 0.95 * torch.min(target_Q_list[0],target_Q_list[1]) + 0.05 *(target_Q_list[2]+target_Q_list[3])/2
			elif algorithm == 'QMD3':
				target_Q_list, indices = torch.sort(target_Q_list,dim=0)
				target_Q = target_Q_list[self.critic_number // 2]

			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		for i in range(self.critic_number):
			current_Q = self.critic[i](state,action)
			critic_loss = F.mse_loss(current_Q,target_Q)

			self.critic_optimizer[i].zero_grad()
			critic_loss.backward()
			self.critic_optimizer[i].step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			actor_loss = 0
			# Compute actor losse
			for i in range(self.critic_number):
				actor_loss = actor_loss - self.critic[i](state, self.actor(state)).mean()
			actor_loss = actor_loss / self.critic_number

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for i in range(self.critic_number):
				for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
