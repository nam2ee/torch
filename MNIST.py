import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        # 입력층에서 은닉층으로 가는 가중치
        self.hidden = nn.Linear(in_features=784, out_features=128)
        # 은닉층에서 출력층으로 가는 가중치
        self.output = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 은닉층에서의 활성화 함수 적용
        x = F.relu(self.hidden(x))
        # 출력층에서의 활성화 함수 적용
        x = F.log_softmax(self.output(x), dim=1)
        return x



# 데이터 로더 설정
# train_loader는 학습 데이터셋을 제공하는 DataLoader 인스턴스입니다.

# 모델 인스턴스 생성
model = SimpleNeuralNetwork()

# 손실 함수 선택
criterion = nn.CrossEntropyLoss()

# 옵티마이저 선택, 여기서는 SGD를 사용합니다.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 에폭 수 설정
epochs = 5

for epoch in range(epochs):  # 전체 데이터셋을 여러 번 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 입력 데이터를 받습니다.
        inputs, labels = data

        # 매 반복마다 이전에 계산된 그라디언트를 초기화합니다.
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 실행합니다.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # 그라디언트 역전파 계산
        optimizer.step() # 역전파를 통해서 가중치 업데이트

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # 매 2000 미니배치마다 출력합니다.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

