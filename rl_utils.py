import math
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import ddpg

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size # type: ignore
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_off_policy_agent(env, agent:ddpg.DDPG, num_episodes, replay_buffer, minimal_size, batch_size):

    rl_f = open('rl_log.csv', 'w')
    cl_f = open('c_loss.csv', 'w')
    al_f = open('a_loss.csv', 'w')
    pi_f = open('policy.csv', 'w')

    def file_log(f, *msg):
        s = ','.join(str(m) for m in msg) + '\n'
        f.write(s)
        f.flush()

    return_list = []
    best_reward = -math.inf
    best_policy = []

    num_step = 0

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                T = []
                while not done:
                    action = agent.take_action(state, num_episodes/10 * i + i_episode+1)
                    next_state, reward, done, _ = env.step(action)
                    # replay_buffer.add(state, action, reward, next_state, done)
                    T.append((state, action, reward, next_state, done))
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        c_loss, a_loss = agent.update(transition_dict)

                        file_log(al_f, f'{num_step}', f'{a_loss.item()}')
                        file_log(cl_f, f'{num_step}', f'{c_loss.item()}')
                        num_step += 1
                
                final_reward = T[-1][2]
                for state, action, reward, next_state, done in T:
                    replay_buffer.add(state, action, final_reward, next_state, done)
                
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_policy = env.strategy

                return_list.append(episode_return)
                
                file_log(rl_f, f'{i*10+i_episode}', f'{episode_return}')
                file_log(pi_f, f'{i*10+i_episode}', f'{best_policy}', f'{env.strategy}')

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list, best_reward, best_policy, env.reward(), env.strategy 

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)