from env import DemoStepEnv, QuantPruneEnv
import ddpg, ddpg
import torch
import random
import numpy as np
import rl_utils
import args
from model import get_vit

if __name__ == '__main__':
    # import wandb
    # wandb.init()
    
    vit_path = r'/home/ma-user/work/Vision-Transformer-ViT/output/mvit/pat-0.5.pt'
    num_episodes = 20000
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 128
    device = 'npu'
    warmup = 50

    lat_b = 1000 #ms
    e_b = 1000 #mj
    mem_b = 70*1024 #KB

    env_name = 'Quant Prune Env'
    env = QuantPruneEnv(
        get_vit(args.MQVIT, vit_path), **args.ENV, lat_b=lat_b, e_b=e_b, mem_b=mem_b, device=device
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dim = env.state_dim
    action_dim = 1
    action_bound = 1  # 动作最大值
    
    agent = ddpg.DDPG(state_dim, action_dim=action_dim, **args.AGENT, device=device, warmup=warmup)

    return_list, r, p, c_r, c_p = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)