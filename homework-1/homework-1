def reverse_double(input:list)->list:
  input = [element * 2 for element in input]
  return [element for element in reversed(input)]

A = [1,2,3,4,5,6]
print(reverse_double(A))



import numpy as np
import matplotlib.pyplot as plt

D = np.random.normal(loc=0, scale=1, size=(1000,1))
D = np.reshape(D, (500,2))

plt.scatter(D[:,0], D[:,1])
plt.title("Scatter Plot for Normal Distribution")
plt.xlabel("x coordinates")
plt.ylabel("y coordinates")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.figure(figsize=(12,12))

plt.show()



import numpy as np
from scipy.sparse import csr_matrix

X = np.random.uniform(low=0, high=1, size=(100,50))
X[X < 0.9] = 0
X_sparse = csr_matrix(X)

print(f'X has type {type(X)} and has {100-np.sum(X!=0)/50}% of zeros')
print(f'X_sparse has type {type(X_sparse)} and has {100-np.sum(X_sparse!=0)/50}% of zeros')



def power_iter(X, num_iter:int):   
  v = np.random.randn(X.shape[1])
  one_vec = np.ones_like(v)
  mu_col_matrix = np.mean(X, axis=1)  # Returns a 1 column matrix since X is of "matrix" type 
  mu = np.array(mu_col_matrix).squeeze()  # Convert from column matrix to 1D array
  
  vec_list = [None] * 1000
  vec_list[0] = np.random.normal(loc=0, scale=1, size=(50,))
  for i in range(1,num_iter-1):
    vec1 = X.T.dot(X.dot(vec_list[i-1]))
    vec2 = one_vec.dot(mu.T.dot(X.dot(vec_list[i-1])))
    vec3 = (X.T.dot(mu)).dot(one_vec.T.dot(vec_list[i-1]))
    vec4 = (one_vec.dot(mu.T.dot(mu))).dot(one_vec.T.dot(vec_list[i-1]))
    vec = vec1 - vec2 - vec3 - vec4
    vec_norm = np.linalg.norm(vec)
    vec_list[i] = vec / vec_norm
  return vec_list[i]

power_iter(X_sparse,1000)
v1_yours = power_iter(X_sparse,1000).squeeze()
print(v1_yours.shape)



def verify_v1(X):
  mu_col_matrix = np.mean(X, axis=1)
  mu = np.array(mu_col_matrix).squeeze()
  Xc = (X.T - mu.T).T
  U, S, VT = np.linalg.svd(Xc)
  vec = VT[-1]
  return vec

# Note here we just pass in the dense 2D array `X`
#  which represents the same matrix as `X_sparse`
v1_simple = verify_v1(X).squeeze()
# Compute a sign corrected difference between the vectors
#  (accounting for the fact that SVD is only unique up to signs)
diff_sign_corrected = np.sign(v1_yours[0]) * v1_yours - np.sign(v1_simple[0]) * v1_simple
mae_corrected = np.mean(np.abs(diff_sign_corrected))
print(f'The average absolute difference of the two function output is {mae_corrected}')
