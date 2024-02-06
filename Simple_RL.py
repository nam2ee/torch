def get_action_value(A):
    return A[1]


def get_best_action(actions):
    best_action = 0
    max_action_value = 0
    for i in range(len(actions)):
        current_value = get_action_value(actions[i])
        if current_value > max_action_value:
            best_action = i
            max_action_value = current_value
    return best_action


def get_reward(prob, n=10):
    reward = 0;
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n = 10
probs = np.random.rand(n) # fixed at once
eps = 0.1 #probability of 일탈

reward_test = [get_reward((probs[1])) for _ in range(2000)]
print(int(np.mean(reward_test)))

record = np.zeros((n,2))
# 0 index에는 레버를 당긴 횟수,  1 index에는 평균값

def get_best_arm(record): #지금 껏 봤을 때, 최적의 레버를 당긴다
    arm_index = np.argmax(record[:,1],axis=0)
    return arm_index


def update_record(record,action,r): # action은 index, r은 result
    new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1)
    record[action,0] += 1
    record[action,1] = new_r
    return record

fig,ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9,5)
rewards = [0]
for i in range(500):
    if random.random() > 0.2:
        choice = get_best_arm(record)
    else: #기하적 확률로 인한 일탈
        choice = np.random.randint(10)
    print(choice)
    r = get_reward(probs[choice])
    record = update_record(record,choice,r)
    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)
    rewards.append(mean_reward)
print(record)
ax.scatter(np.arange(len(rewards)),rewards)


def softmax(av, tau=1.12):
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm #타우가 커지면 확률들이 서로 비슷해진다!