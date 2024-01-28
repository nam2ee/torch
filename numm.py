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


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)" # special extract!
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)" # extract 1~2 as part of instance

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[[1,1,1],[0,0,0]]) # 고급 인덱싱 -> 각 배열이 행, 열을 나타낸다!


a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # np arrange(4) -> make [0,1,2,3]

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])

print(a>2) #원소 각각에 대해서 수행

#1,2       5,6      -> 19 22
#3,4       7,8         43 50


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

print(np.multiply(x,y)) # elementwise- multiply
print(np.dot(x, y))  # matrix product

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)

for i in range(4):
    y[i,:] = x[i,:] + v
print(y)

vv = np.tile(v,(4,2)) ## 4,2의 타일로 v를~
print(vv)#[[1 0 1 1 0 1]
         #[1 0 1 1 0 1]
         #[1 0 1 1 0 1]
         #[1 0 1 1 0 1]]

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)


# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w) # reshape != tile


