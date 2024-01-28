import numpy as np

a = np.array([1,2,3])
print(type(a))
print(a.shape)

print(a[0],a[1],a[2])

b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0,0],b[0,1],b[0,2])

a= np.zeros((2,2))
print(a)

d = np.eye(2) # identity matrix
print(d)