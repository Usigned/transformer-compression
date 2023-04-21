import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import args
from model import get_vit

from env import QuantPruneEnv
from ddpg import DDPG

if __name__ == '__main__':
    import torch_npu

    env_log_path = 'env.log'


    num_episodes = 5000
    minimal_size = 1000
    batch_size = 64
    buffer_size = 10000

    env_name = 'MyEnv'
    env = QuantPruneEnv(get_vit(args.MQVIT), **args.ENV)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.state_dim
    action_dim = 1
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim, **args.AGENT 
    )

    return_list, best_reward, best_s, r, s  = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)