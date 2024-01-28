import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")

device = torch.device("mps")
print(device)

a = np.ones((2,3), dtype = "float32")
print(a.dtype, a.shape)

b = torch.tensor(a)
print(b.dtype, b.shape)

x = torch.tensor([[1,2,3],[4,5,6] ], dtype = torch.int32, device=device)
print(x.device, x.dtype)

a= torch.tensor([1,2,3,4,6])
print(a[2],a[4].item())
print(a.dtype)
b = a.to(device)
print(b)

x = torch.rand(2,3)
x = x.to(device)
print(x.shape,x.ndimension())
a = a.view(5,1) # view -> 효율적인 연산: 데이터 안 바꾸고 그대로!
                       # 데이터 안 바꾸고 새로운 관점 (view)로 보는 것
print(a)
print(a.reshape(1,5)) # reshape-> 완전히 메모리를 새로 복사하므로,
                      # non-contiguous하게 될 수 있다!

a = torch.rand(4,4)
print(a.is_contiguous())# 1234 /5678/ ...가로로 정렬
a = a.transpose(0,1)
print(a.is_contiguous()) #데이터 저장방향이 바뀌어서,,,
print(a.stride()) # (1,4) means -> a[1][0] - 1 change / a[0][1] -> 4 change
b = a.view(-1,1) #
print(b.is_contiguous())
a = a.transpose(0,1) #transpose axis
print(a)
print(a.is_contiguous())
print(a.reshape(-1,1))