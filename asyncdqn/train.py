#coding=utf-8
import gym
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn import init
from torch.autograd import Variable
from keras.utils import to_categorical

gamma = 0.95
max_expor = 1.
min_expor = 0
target_update_freq = 8000
nb_steps = 500

class QModel(nn.Module):
	def __init__(self, state_space, nb_action, dueling=True):
		super(QModel, self).__init__()
		self.state_space = state_space
		self.nb_action = nb_action
		self.dense1 = nn.Linear(state_space, 64)
		self.dense2 = nn.Linear(64, 32)
		if dueling:
			self.dense_advantage = nn.Linear(32, nb_action)
			self.dense_v = nn.Linear(32, 1)
		self.dense_q = nn.Linear(32, nb_action)
		self._init_weights()
		self.dueling = dueling
        
	def forward(self, state, action):
		x = F.relu(self.dense1(state))
		x = F.relu(self.dense2(x))
		if self.dueling:
			self.advantage_q = (self.dense_advantage(x)).view(-1, self.nb_action)
			self.v = (self.dense_v(x)).view(-1, 1)
			q_val = self.advantage_q + self.v - self.advantage_q.mean(dim=1, keepdim=True)
		else:
			q_val = self.dense_q(x)
		self.q_values = q_val
		q_hat = q_val.mul(action).sum(-1, keepdim=True)
		return q_hat
    
	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal(m.weight.data)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				init.xavier_normal(m.weight.data)
				m.bias.data.fill_(0)

def state_to_tensor(state, var = True):
	_state = torch.FloatTensor(state).unsqueeze(0)
	return Variable(_state) if var else _state

def action_onehot_tensor(action, nb_action, var=True):
	action_one_hot = to_categorical(action, nb_action)
	_tensor = torch.FloatTensor(action_one_hot)
	return Variable(_tensor) if var else _tensor

def eps_greedy(model, eps, state, nb_action):
	if random.random() < eps:
		return np.random.choice(nb_action)
	fake_action = action_onehot_tensor(0, nb_action)
	_ = model(state, fake_action)
	q_values = model.q_values.data
	return q_values.max(-1)[1][0]

def softmax_policy(model, state, nb_action):
	fake_action = action_onehot_tensor(0, nb_action)
	_ = model(state, fake_action)
	q_values = model.q_values.view(-1, nb_action)
	soft_q = F.softmax(q_values, dim=1).data.numpy()
	return np.random.choice(nb_action, p=soft_q[0])	

def copy_to_target(model, target):
	target.load_state_dict(model.state_dict())
	target.eval()
	for param in target.parameters():
		target.requires_grad = False

def moving_copy_to_target(model, target, tau=0.95):
	for p, tp in zip(model.parameters(), target.parameters()):
		tp.data = (1 - tau ) * p.data + tau * tp.data

def copy_shared_grads(model, shared_model, counter):
	for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
		if param.grad is not None:
			shared_param.grad = (param.grad.clone())

def train(rank, shared_model, target_model, counter, lock, args, optimizer):
    #根据A3C paper 每个进程有自己的env
	env = gym.make('CartPole-v0')
	nb_action = env.action_space.n
	state_space = env.observation_space.shape[0]
	torch.manual_seed(args.seed + rank) #确保每个线程的随机数种子不一样
	random.seed(args.seed + rank)
	np.random.seed(args.seed + rank)
	copy_interval = rank % 4 + 1 
	if optimizer is None:
		optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)
	model = QModel(state_space, nb_action, args.dueling) #用于更新梯度的模型
	model.train()
	#crit = nn.SmoothL1Loss()
	crit = nn.MSELoss()
	_eps_step = (max_expor - min_expor) / nb_steps
	pbar = tqdm(range(nb_steps))
	avg_reward = 0
	reward_his = np.empty(nb_steps)
	for i in pbar:
		with lock:
			if i % copy_interval == 0:
				model.load_state_dict(shared_model.state_dict()) #每次完整经历使用最新的主模型的参数
		model.zero_grad()
		done = False
		total_reward = 0
		eps = max(max_expor - _eps_step * i, min_expor) 
		state = state_to_tensor(env.reset())
		loss, reward, _step = 0, 0, 0
		while not done:
			_step += 1
			action = eps_greedy(model, eps, state, nb_action)  #从最新的模型中用epislon-greedy选择一个动作
			#action = softmax_policy(model, state, nb_action)
			action_one_hot = action_onehot_tensor(action, nb_action)
			next_state, reward, done, _ = env.step(action)
			next_state = state_to_tensor(next_state)
			total_reward += reward
			if done:
				reward = -1
			with lock:
				counter.value += 1 #总动作数
            #y_target = r + gamma * Q_target(s',argmax Q(s',a)) DoubleDQN 用于改善OverEstinmate 
			max_action = action_onehot_tensor(eps_greedy(model, -1, next_state, nb_action), nb_action)
			#获取s'的最大动作，DoubleDQN
			dw = 0 if done else 1
			q_pred = target_model(next_state, max_action)
			y_target = reward + gamma * dw * q_pred.detach()
			model_output = model(state, action_one_hot)
			loss = crit(model_output, y_target)
			loss.backward() #积累梯度
			state = next_state
			with lock:
				#if counter.value % target_update_freq == 0:
					#copy_to_target(shared_model, target_model)
            	#每隔一段时间拷贝主模型参数给target. target并不进行梯度更新，仅仅用于延迟预测Q值 
				moving_copy_to_target(shared_model, target_model, tau=0.95)
        #梯度剪裁
		nn.utils.clip_grad_norm(model.parameters(), 10.)
		with lock:
			optimizer.zero_grad()
			copy_shared_grads(model, shared_model, counter)
        	#将累计梯度更新到主模型中 然后更新主模型, 要求主模型是shared_memory
			optimizer.step()
		reward_his[i] = total_reward 
		avg_reward = 0.8 * avg_reward + 0.2 * total_reward
		#if avg_reward >= 190:
		#	break
		pbar.set_description('Thread {} Reward:{} Loss:{:.4f} Avg:{:.2f}'.format(rank,
                total_reward, loss.data[0], avg_reward))
	if args.save:
		torch.save(reward_his, open('rd{}.pkl'.format(rank), 'wb'))
