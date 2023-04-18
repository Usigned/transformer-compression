import math
from data import get_cifar10_dataloader
from ddpg import DDPG
from env import Env, DemoStepEnv
from copy import deepcopy
import args
from model import get_vit
import torch
# import wandb

# wandb.init()
def train_agent(num_episode, env:DemoStepEnv, agent:DDPG, args, output=None):
    
    best_reward = -math.inf
    best_policy = []

    agent.is_training = True
    state = None

    step = episode = episode_steps = 0
    episode_reward = 0.

    T = []  # trajectory

    while episode < num_episode:

        if state is None:
            state = deepcopy(env.reset())
            agent.reset(state)

        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(state, episode=episode)
            # wandb.log({'action': action})

        state2, reward, done, info = env.step(action)
        state2 = deepcopy(state2)

        T.append([reward, deepcopy(state), deepcopy(state2), action, done])

        # [optional] save intermideate model
        if output:
            if episode % int(num_episode / 10) == 0:
                agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        state = deepcopy(state2)

        if done:  # end of episode
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()


            # reset
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy

    return best_policy, best_reward

def eval_agent(env:DemoStepEnv, agent:DDPG):
    agent.is_training = False
    state = None
    done = False
    info = {}

    for i in range(2):
        while not done:
            if state is None:
                state = deepcopy(env.reset())
                agent.reset(state)
            action = agent.select_action(state, episode=500)
            print(action, agent.critic((torch.tensor([state], dtype=torch.float), torch.tensor([action], dtype=torch.float))))
            state, reward, done, info = env.step(action)


        print(info['info'])

if __name__ == '__main__':
    # path = r'/Users/qing/Downloads/0.9853000044822693.pt'
    # trainloader = get_cifar10_dataloader()
    # testloader = get_cifar10_dataloader(train=False)
    # env = Env(get_vit(args.QVIT), path, trainloader,
    #           testloader, 'cpu', args.ENV)

    env = DemoStepEnv()
    nb_states = env.len+1
    nb_actions = 1
    agent = DDPG(nb_states, nb_actions, args.TRAIN_AGENT, 'cpu')
    p, r = train_agent(300, env, agent, args.TRAIN_AGENT)
    agent.save_model('.')
    print(p, r)

    # agent.load_weights('.')
    # eval_agent(env, agent)
    # print(env._action_wall(agent.actor(torch.tensor([1.]*7))))
