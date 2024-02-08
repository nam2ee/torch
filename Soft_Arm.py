import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


# Step 1. explore - Here is chosen by prob only!!!
# Step 2. exploit


# Environment
# 1. How to score?
# 2. How to

# 절차
#  탐색한다: 무작위 팔을 뽑는다  - 스코어링을 통해 점수를 받는다 - 업데이트한다  / 활용한다: 최고의 팔을 뽑는다 - 스코어링을 통해 점수를 받는다 - 업데이트한다
#  불확실한 환경에서 나름 나쁘지 않은 방법

hyperparm_epsilon = 0.2
hyperparm_num = 10
record = np.zeros((hyperparm_num,3))
probs = np.zeros(hyperparm_num)

# record[i,0]는 확률
# record[i,1]는 평균점수

for i in range(hyperparm_num):
    record[i,0] = random.random()


def softmax(av, tau=1.12):
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm #타우가 커지면 확률들이 서로 비슷해진다!

def get_best_arm():
    pick = 0
    max_value = 0
    for i in range(hyperparm_num):
        if(record[i,1] > max_value):
            max_value = record[i,1]
            pick = i
        else:
            continue
    return pick


def scoring(random_num):
    reward = 0;
    for i in range(hyperparm_num):
        if random.random() < random_num:
            reward += 1
    return reward


def update(score:int, arm_num:int):
    record[arm_num,0] = (record[arm_num,1]* record[arm_num,2] + score) / (record[arm_num,2]+1)
    record[arm_num,2] += 1


#main logic

sum = 0

for i in range(500):
    choice = get_best_arm()
    ran = random.random
    score = scoring(record[choice,0])
    sum += score
    update(score, choice)
print(sum)
