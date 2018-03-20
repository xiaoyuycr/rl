import gym
import torch
import torch.multiprocessing as mp
from collections import namedtuple
from my_optim import AsyncRMSprop
from train import *

counter = mp.Value('i', 1)
lock = mp.Lock()
save, dueling = True, True
lr = 0.001
arg_list = namedtuple('args', 'lr gamma seed save dueling')
seed = 1
q_args = arg_list(lr, gamma, seed, save, dueling)
processes = []
num_process = 8

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    nb_action = env.action_space.n
    state_space = env.observation_space.shape[0]
    torch.manual_seed(seed)
    target_qmodel = QModel(state_space, nb_action, dueling)
    target_qmodel.eval()
    for param in target_qmodel.parameters():
        param.requires_grad = False
    target_qmodel.share_memory()
    shared_q_model = QModel(state_space, nb_action, dueling)
    shared_q_model.share_memory()
    shared_q_model.train()
    optimizer = AsyncRMSprop(shared_q_model.parameters(), lr=lr)
    for rank in range(num_process):
        p = mp.Process(
            target=train,
            args=(rank, shared_q_model, target_qmodel, counter, lock, q_args, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    if save:
        torch.save(shared_q_model.state_dict(), open('asdqn.pkl', 'wb'))
