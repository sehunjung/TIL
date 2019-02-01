
# import numpy as np
# import matplotlib.pylab as plt
# # def AND(x1, x2):
# #     w1, w2, theta = 0.5, 0.5, 0.7
# #     tmp = x1*w1 + x2*w2
# #     if tmp <= 0.5:
# #         return 0
# #     elif tmp > theta:
# #         return 1

# # print(AND(0,0))
# # print(AND(1,0))
# # print(AND(0,1))
# # print(AND(1,1))

# def AND(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])
#     b = -0.7
#     tmp = np.sum(w*x) + b

#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# # print(AND(0,0))
# # print(AND(1,0))
# # print(AND(0,1))
# # print(AND(1,1))        

# def NAND(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7

#     tmp = np.sum(w*x) + b

#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# def OR(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])
#     b = -0.2

#     tmp = np.sum(w*x) + b

#     if tmp <= 0:
#         return 0
#     else:
#         return 1    

# def XOR(x1, x2):
#     s1 = NAND(x1, x2)
#     s2 = OR(x1, x2)
#     y = AND(s1, s2)
#     return y

# print(XOR(0,0))
# print(XOR(1,0))
# print(XOR(0,1))
# print(XOR(1,1))     

# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)

import matplotlib.pylab as plt
import numpy as np

# def step_function(x):
#     return np.array( x > 0, dtype=np.int)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

X = np.array([1,2])
print(X.shape)