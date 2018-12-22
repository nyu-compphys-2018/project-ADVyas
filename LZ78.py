import numpy as np
import math

def compress(uncompressed):
    """Compress a string to a list of output symbols."""

    # Build the dictionary.
    ds = 0
    dictionary = []
    d1 = []
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

def decompress(compressed):
    file = []
    w = []
    dictionary = []
    for i in range(len(compressed)):
        if compressed[i][0] < 1:
            dictionary.append([compressed[i][1]])
        else:
            w = dictionary[compressed[i][0]-1] + [compressed[i][1]]
            dictionary.append(w)
            w = []
    print(dictionary)
    for i in range(len(dictionary)):
        file += dictionary[i]
    return file
        
    

a = np.array([0, 1, 1, 109,30,43,12,12,12, 1,21,21,12,32,41,145,11,21, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,0])
str1 = "".join(str(e) for e in a)
b ="ABCABBBAAABACXZASIZgfhgkkghffkfkfkfkfftawsesdfbhhgnANIAND"
l =len(a)
print(a)
b1 = compress(a)
print(b1)
print(len(b1))
C =float(len(b1))
CID = (C*math.log(C,2) + C*math.log(C,2))/l
print(CID)
print(np.array(decompress(b1)))
