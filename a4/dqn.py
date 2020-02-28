import gym
import math
import numpy as np
import os
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer

USE_MEMORY = True
DOUBLE_Q = True


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.15 if USE_MEMORY else 0.3
TARGET_UPDATE = 25
NUM_EPISODES = 4000
TEST_INTERVAL = 200
PRINT_INTERVAL = 50
LEARNING_RATE = 1e-5
if USE_MEMORY:
    LEARNING_RATE = (4e-4 if DOUBLE_Q else 3e-3)
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.MSELoss(reduction='mean')
memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    if random.random() < EPS_EXPLORATION:
        return env.action_space.sample()
    return model.select_action(torch.from_numpy(state)).item()

# Given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
def optimize_model(state, action, next_state, reward, done):
    # For double Q, use the other Q function for getting values of the next state.
    nextStateQ = target if DOUBLE_Q else model

    if not USE_MEMORY:
        y_j = reward
        if not done:
            y_j += GAMMA * nextStateQ(torch.from_numpy(next_state)).max()

        y_j_torch = torch.FloatTensor([y_j])
        y_pred = model(torch.from_numpy(state))[action]

    else:
        memory.push(state, action, next_state, reward, done)
        stateBatch, actionBatch, next_stateBatch, rewardBatch, doneBatch = memory.sample(BATCH_SIZE)

        y_j_torch = rewardBatch + (1 - doneBatch) * (GAMMA * nextStateQ(next_stateBatch).max(1)[0])
        y_pred = torch.squeeze(torch.gather(model(stateBatch), 1, actionBatch), dim=1)

    # Gradient descent on MSE loss between predicted and calculated Q
    loss = loss_function(y_j_torch, y_pred)
    lossValue = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return lossValue

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

            loss = optimize_model(state, action, next_state, reward, done)

            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if (i_episode % PRINT_INTERVAL == 0):
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}] - loss {:.2f}'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward, loss))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render, verbose=False)
            if score > best_score:
                best_score = score
                # Force directory to be same as this file.
                fileName = "best_model_{}.pt".format(ENV_NAME)
                fullPath = os.path.join(os.path.dirname(__file__), fileName)
                torch.save(model.state_dict(), fullPath)
                print('saving model to %s' % fullPath)
            print("[TEST Episode {:4d}] [Average Reward {:.1f} vs best {:.1f}]".format(i_episode, score, best_score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
