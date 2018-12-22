import numpy as np

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

print(Hilbert(3,0,3))

w = np.zeros([64,64])
for k in range (64):
    for i in range(64):
        w[-i-1,k] = Hilbert(6,k,i)
print(w)
        
