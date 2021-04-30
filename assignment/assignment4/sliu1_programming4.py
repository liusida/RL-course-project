import numpy as np
import random
import gym
from collections import defaultdict
import gymgrid
env = gym.make('cliff-v0')

# parameters
gamma = 0.5 # discounting rate
alpha = 0.5 # (0,1] // stepSize
epsilon = 0.1 # e-greedy
# Q-values
Q = defaultdict(lambda:  defaultdict(lambda: 0.0))
# Rollout
def run(training=True, random_agent=False, hand_designed=[], render=True):
    global Q
    state = env.reset()
    rewards = []
    actions = []
    for i in range(1000):
        if render:
            env.render()
        if len(hand_designed)>0: # hand designed
            action = hand_designed[i]
        elif (not training or (random.random() > epsilon and len(Q[state])>0)) and not random_agent: # be greedy
            action = max(Q[state].items(), key=lambda a: a[1])[0]
        else:
            action = env.action_space.sample()
            # print(f"random action {action}")
        actions.append(action)
        old_state = state
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
        if training:
            Q[old_state][action] += alpha*(reward + gamma*Q[state][action] - Q[old_state][action])
    return np.sum(rewards)
# train
for it in range(100):
    epsilon *= 0.9
    ret = run(training=True, render=False)
    print(f"Iteration {it}, epsilon {epsilon}: Return {ret}.")
# test
ret = run(training=False, render=True)
print(f"Test: Return {ret}.")
ret = run(random_agent=True, render=False)
print(f"Random: Return {ret}.")
hand_designed = [2] + [1]*11 + [3]
ret = run(hand_designed=hand_designed, render=True)
print(f"Hand Designed: Return {ret}.")
env.close()
print(f"Total possible state: {len(Q)}")
