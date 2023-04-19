from env import DemoStepEnv
import ddpg, ddpg
import torch
import random
import numpy as np
import rl_utils
import args
import memory

if __name__ == '__main__':
    import wandb
    wandb.init()
    
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 10000
    hidden_dim = 64
    gamma = 1.0
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.03  # 高斯噪声标准差
    init_delta = 0.7
    delta_decay= 0.995
    warmup_turns = 20
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Demo Step Env'
    env = DemoStepEnv(20, 1, 21, list(range(1, 21)))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    # replay_buffer = memory.SequentialMemory(buffer_size, window_length=1)

    state_dim = env.len+1
    action_dim = 1
    action_bound = 1  # 动作最大值
    
    agent = ddpg.DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, warmup=warmup_turns, init_delta=init_delta, delta_decay=delta_decay, device=device)

    return_list, r, p, c_r, c_p = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    print(r, p)
    print(c_r, c_p)