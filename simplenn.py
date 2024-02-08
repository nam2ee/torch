import torch

random_tensor = torch.rand(2,3)
addded_tensor = random_tensor+5

reshaped_tensor = addded_tensor.view(3,2)

# 1. 값이 2인 스칼라 텐서 생성, requires_grad=True로 설정하여 자동 미분 활성화
x = torch.tensor(2.0, requires_grad=True)

# 2. y = x^2 연산 수행
y = x ** 2

# 3. backward 함수를 호출하여 미분 계산
y.backward()
print(y)
import torch.nn as nn
import torch.nn.functional as F

# 신경망 클래스 정의
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        # 입력층에서 은닉층으로 가는 가중치
        self.hidden = nn.Linear(1, 1)
        # 은닉층에서 출력층으로 가는 가중치
        self.output = nn.Linear(1, 1)

    def forward(self, x):
        # 은닉층을 통과한 후 시그모이드 활성화 함수 적용
        x = torch.sigmoid(self.hidden(x))
        # 출력층을 통과
        x = self.output(x)
        return x

# 신경망 인스턴스 생성
net = SimpleNeuralNetwork()

# 임의의 입력 텐서 생성
input_tensor = torch.tensor([1.0])

# 순전파 실행
output = net(input_tensor)
print(output)