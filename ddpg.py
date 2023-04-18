from collections import deque
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque, namedtuple
from random import sample

Experience = namedtuple(
    'Experience', 'state0, action, reward, state1, terminal1')


def to_numpy(var: torch.Tensor):
    return var.cpu().data.numpy()


def to_tensor(ndarray: np.ndarray, device, volatile=False, requires_grad=False, dtype=torch.FloatTensor):
    return Variable(
        torch.from_numpy(ndarray).to(device), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = self.fc11(x) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args, device):

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        # Make sure target is with the same weight
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = VarBatchSizeMemory(limit=args.rmsize)

        # Hyper-parameters
        # self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.warmup = args.warmup
        self.delta = args.init_delta
        # loss
        self.value_loss = 0.0
        self.policy_loss = 0.0

        #
        self.epsilon = 1.0
        # self.s_t = None  # Most recent state
        # self.a_t = None  # Most recent action
        self.is_training = True

        self.device = device
        self.to(self.device)

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

        self.criterion = nn.MSELoss()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_and_split()

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * \
                (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average
        # if reward_batch.std() > 0:
        #     reward_batch /= reward_batch.std()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch, device=self.device),
                self.actor_target(
                    to_tensor(next_state_batch, device=self.device)),
            ])

        target_q_batch = to_tensor(reward_batch, device=self.device) + \
            self.discount * \
            to_tensor(terminal_batch.astype(np.float32),
                      device=self.device) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch, device=self.device), to_tensor(
            action_batch, device=self.device)])

        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch, device=self.device),
            self.actor(to_tensor(state_batch, device=self.device))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        # update for log
        self.value_loss = value_loss
        self.policy_loss = policy_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def to(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

    def observe(self, r_t, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, s_t1, done)  # save to memory
            # self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        # self.a_t = action
        return action

    def select_action(self, s_t, episode, decay_epsilon=True):
        # assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
        action = to_numpy(self.actor(
            to_tensor(np.array(s_t).reshape(1, -1), device=self.device))).squeeze(0)
        delta = self.init_delta * (self.delta_decay ** (episode - self.warmup))
        # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        # from IPython import embed; embed() # TODO eable decay_epsilon=True
        action = sample_from_truncated_normal_distribution(
            lower=self.lbound, upper=self.rbound, mu=action, sigma=delta)
        action = np.clip(action, self.lbound, self.rbound)
        # update for log
        self.delta = delta
        # self.a_t = action
        return action

    def reset(self, obs):
        pass
        # self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_delta(self):
        return self.delta

    def get_value_loss(self):
        return self.value_loss

    def get_policy_loss(self):
        return self.policy_loss


class VarBatchSizeMemory:
    def __init__(self, limit):
        # batch_size, state0, action, reward, state1, terminal
        self.deque = deque([], limit)
        self.cur_batch = []

    def append(self, state0, action, reward, state1, terminal, training=True):
        if training:
            self.cur_batch.append(Experience(
                state0, action, reward, state1, terminal))

    def make_batch(self):
        if len(self.cur_batch) != 0:
            self.deque.append(self.cur_batch)
            self.clear_cur_batch()
            return
        raise RuntimeError('Empty Cur Batch')

    def sample(self):
        return sample(self.deque, 1)

    def sample_and_split(self):
        '''
        s0,a,r,s1,t1
        '''
        experiences = self.sample()[0]

        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch, 'double')
        state1_batch = np.array(state1_batch, 'double')
        terminal1_batch = np.array(terminal1_batch, 'double')
        reward_batch = np.array(reward_batch, 'double')
        action_batch = np.array(action_batch, 'double')

        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

    def has_data(self):
        return self.num_batches != 0

    def reset(self):
        self.deque.clear()

    def clear_cur_batch(self):
        del self.cur_batch
        self.cur_batch = []

    @property
    def num_batches(self):
        return len(self.deque)
