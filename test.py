import numpy as np
import math

tb1 = np.ones((1,1,32))
print(tb1)
'''
for i in range(1):
    for j in range(1):
        print(tb1[i,j])
prob_c = [0.0 for i in range(10)]
print(prob_c)
a = np.array([8,2,3,4,5])
b = np.argmin(a)
print(a[b])
'''
a_list = [1,1,1,0,0]

correct  = 0
wrong = 0
for i in range(len(a_list)):
    if a_list[i] == 1:
        correct+=1
    else:
        wrong+=1
 
print(correct)





data_list = [[[] for i in range(5)] for j in range(10)]
print(data_list)

print(math.factorial(3))
