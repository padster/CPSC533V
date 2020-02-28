import gym
import math
import numpy as np
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 30
NUM_EPISODES = 4000
TEST_INTERVAL = 250
PRINT_INTERVAL = 50
LEARNING_RATE = 1e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions, epsGreedy=EPS_EXPLORATION, randomActionFunc=env.action_space.sample).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.MSELoss(reduction='sum')
memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    if random.random() < EPS_EXPLORATION:
        return env.action_space.sample()
    return model.select_action(torch.from_numpy(state)).item()

def optimize_model(state, action, next_state, reward, done, useMemory=True, doubleQ=True):
    nextStateQ = target if doubleQ else model

    # Given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    if not useMemory:
        y_j = reward
        if not done:
            y_j += GAMMA * nextStateQ(torch.from_numpy(next_state)).max()

        y_j_torch = torch.FloatTensor([y_j])
        y_pred = model(torch.from_numpy(state))[action]

        loss = loss_function(y_j_torch, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    else:
        memory.push(state, action, next_state, reward, done)
        stateBatch, actionBatch, next_stateBatch, rewardBatch, doneBatch = memory.sample(BATCH_SIZE)

        y_j_torch = rewardBatch + (1 - doneBatch) * (GAMMA * nextStateQ(next_stateBatch).max(1)[0])
        y_pred = torch.squeeze(torch.gather(model(stateBatch), 1, actionBatch), dim=1)

        loss = loss_function(y_j_torch, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")
    
    for i_episode in range(1, NUM_EPISODES+1):
        state = env.reset()
        episode_total_reward = 0
        for t in count():
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action) #action.cpu().numpy()[0][0])
            steps_done += 1
            episode_total_reward += reward

            optimize_model(state, action, next_state, reward, done)

            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if (i_episode % PRINT_INTERVAL == 0):
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render, verbose=False)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {:4d}] [Average Reward {:.1f} vs best {:.1f}]".format(i_episode, score, best_score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
