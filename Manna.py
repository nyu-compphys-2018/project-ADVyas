import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math

N = 3600
s = 64
A = np.zeros([s,s])
for i in range(N):
    b = np.random.randint(0,s**2 -1)
    c = b//s
    d = b-c*s
    A[c,d-1] += 1
    N -= 1
writer = cv2.VideoWriter('test17.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(s,s))
t = 600
#print(A)

for i in range(t):
    C = np.zeros([s,s])
    for i in range (s):
        for k in range (s):
            if A[i,k] >= 2.0:
                m = A[i,k] + 0
                C[i,k] += -1*m
                for l in range(int(m+0.4)):
                    u = np.random.randint(0,4)
                    #if u == 0:
                    #    print(u)
                    #if u == 3:
                    #    print(u)
                    if u >= 0.9 and u <= 1.1:
                        C[i-1,k] += 1
                    elif u >= 1.9 and u <= 2.1:
                        C[i,(k+1)%s] += 1
                    elif u >= 2.9 and u <= 3.1:
                        C[(i+1)%s,k] += 1
                    elif u <= 0.1:
                        C[i,k-1] += 1
    #print(sum(sum(C)))
    A = A + C
    B = np.zeros([s,s])
    for i in range (s):
        for k in range (s):
            if A[i,k] <= 0.1:
                B[i,k] = int(255)
            elif A[i,k] <= 1.9 and A[i,k] >= 1:
                B[i,k] = int(200)
            else:
                B[i,k] = int(0)
    #plt.imshow(B, cmap="gray")
    #plt.show()
    #plt.pause(1)
    B = B.astype('uint8')
    B = np.repeat(B,3,axis=1)
    B = B.reshape(s, s, 3)
    writer.write(B)
    #print(A)

#D = np.zeros([s,s])

#for i in range (s):
#    for k in range (s):
#        if B[i,k] <= 0.1:
#            D[i,k] = int(-1)
#        else:
#            D[i,k] = int(1)
            
#F = np.kron(np.ones([s//2,s//2]),np.array([[1,-1],[-1,1]]))
#D = D*F

print (sum(sum(A)))
            
cv2.destroyAllWindows()
writer.release()    
plt.imshow(B, cmap="gray")
plt.show()
