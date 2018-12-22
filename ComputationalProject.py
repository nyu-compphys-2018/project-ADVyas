import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import numba

Nt = np.arange(16000,33500,1500)
res = []
res1 = []
for N in Nt:
    s = 128
    A = np.zeros([s,s])
    for i in range(N):
        b = np.random.randint(0,s**2 -1)
        c = b//s
        d = b-c*s
        A[c,d-1] += 1
        N -= 1
    #writer = cv2.VideoWriter('test8.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(s,s))
    t = 12000
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
                B[i,k] = int(0)
            elif A[i,k] <= 1.9 and A[i,k] >= 1:
                B[i,k] = int(0)
            else:
                B[i,k] = int(1)
        #plt.imshow(B, cmap="gray")
        #plt.show()
        #plt.pause(1)
        #B = B.astype('uint8')
        #B = np.repeat(B,3,axis=1)
        #B = B.reshape(s, s, 3)
        #writer.write(B)
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

    #cv2.destroyAllWindows()
    #writer.release()    
    #plt.imshow(B, cmap="gray")
    #plt.show()
    #plt.imshow(D, cmap="gray")
    #plt.show()

    def Hilbert(m,x,y):
        n = 0
        s = 2**m
        u = 1
        v = 1
        d = 0

        while (0 < s):

            if n ==1:
                u = u*(-1)

            elif n ==2:
                y -=  s
                d += ((2*s)**2)/4

            elif n == 3:
                x -= s
                y -= s
                d += ((2*s)**2)/2

            elif n == 4:
                x -= s
                d += 3*((2*s)**2)/4
                v = v*(-1)

            if x < 0:
                x += s
            if y < 0:
                y += s

            if u < 0 and x < s//2 and y >= s//2 and j == 0:
                x -= s//2
                y -= s//2
                j = 1

            if u < 0 and x >= s//2 and y < s//2 and j ==0:
                x -= s//2
                y -= s//2
                j = 1

            if v < 0 and x < s//2 and y < s//2 and j ==0:
                x -= s//2
                y -= s//2
                j = 1

            if v < 0 and x >= s//2 and y >= s//2 and j ==0:
                x -= s//2
                y -= s//2
                j = 1


            if x < 0:
                x += s
            if y < 0:
                y += s

            if x < s//2 and y < s//2:
                n = int(1)
            elif x < s//2 and y >= s//2:
                n = int(2)
            elif x >= s//2 and y >= s//2:
                n = int(3)
            elif x >= s//2 and y < s//2:
                n = int(4)
            j = 0
            s = s//2
            #print(x)
            #print(y)
            #print(n)
            #print(u)
        return d
    def compress(uncompressed):
        """Compress a string to a list of output symbols."""

        # Build the dictionary.
        ds = 0
        dictionary = []
        d1 = []
        # in Python 3: dictionary = {chr(i): chr(i) for i in range(dict_size)}

        w = []

        for c in uncompressed:
            w.append(c)
            wc = list(w)
            if wc in dictionary:
                w = list(wc)
                ds = dictionary.index(wc) + 1
            else:
                # Add wc to the dictionary.
                dictionary.append(wc)
                if len (wc) < 1.5:
                    ds = 0
                d1.append((ds,c))
                ds = 0
                w = []

        if ds >= 0.5:
            d1.append((ds,))
        # Output the code for w.
        #if w:
        #   result.append(dictionary[w])
        return d1 

    ph1 = 0
    ph2 = 0
    p = np.zeros(s**2)
    p1 = np.zeros(s**2)
    w = np.zeros([s,s])
    for i in range (s):
        for k in range(s):
            w[-i-1,k] = Hilbert(7,k,i)
    for i in range (s):
        for k in range(s):
            p[int(w[k,i])] = A[k,i]
            p1[int(w[k,i])] = B[k,i]
            j = i + k
            if j%2 < 0.5 and B[i,k] > 0.5 :
                ph1 += 1
            elif j%2 > 0.5 and B[i,k] > 0.5:
                ph2 += 1
    
    alp = len(set(p))
    alp1 = len(set(p1))
    l =float(len(p))
    l1 = float(len(p1))
    b1 = compress(p)
    b2 = compress(p1)
    C =float(len(b1))
    C1 = float(len(b2))
    CID = (C*math.log(alp,2) + C*math.log(C,2))/l
    CID1 = (C1*math.log(alp1,2) + C1*math.log(C1,2))/l1
    print(CID)
    print(CID1)
    print(ph1-ph2)
    res.append(CID)
    res1.append(CID1)
    
Ns = Nt/float(s**2)
plt.scatter(Ns,res)
plt.xlabel('Rho')
plt.ylabel('CID')
plt.title('Manna Model of 128 x 128')
plt.show()
