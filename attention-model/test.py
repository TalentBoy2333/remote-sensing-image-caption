import numpy as np 
a = [1,2,3,4]
b = a+[5,6,7,8]
print(b)
b = np.array(b)
print(b.mean())