from env import DemoStepEnv, QuantPruneEnv, CombQuantPruneEnv
import ddpg, ddpg
import torch
import random
import numpy as np
import rl_utils
import args
from model import get_vit

if __name__ == '__main__':
    
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

    lambda_acc = 20
    lambda_lat = 1e-3
    lambda_e = 1e-3
    lambda_mem = 1e-5


    env_name = 'Quant Prune Env'
    # env = QuantPruneEnv(
    #     get_vit(args.MQVIT, vit_path), **args.ENV, lat_b=lat_b, e_b=e_b, mem_b=mem_b, device=device
    # )


    env = CombQuantPruneEnv(
        get_vit(args.MQVIT, vit_path), **args.ENV, device=device, lambda_acc=lambda_acc, lambda_lat=lambda_lat, lambda_mem=lambda_mem, lambda_e=lambda_e
    )

    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dim = env.state_dim
    action_dim = 1
    action_bound = 1  # 动作最大值
    
    agent = ddpg.DDPG(state_dim, action_dim=action_dim, **args.AGENT, device=device, warmup=warmup)

    return_list, r, p, c_r, c_p = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)