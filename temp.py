import multiprocessing
import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt



def argmax(x, key):
    return max(enumerate(x), key=lambda x: key(x[1]))[0]

if __name__ == "__main__":

    N = 5000
    C = 22
    points = np.random.random((N,2)) # points inside (1,1)

    points = points[np.argsort(points[:,0])] # sort by x

    count = lambda p: len(points[
            (points[:,0] <= p[0]) &
            (points[:,1] <= p[1])
    ])

    interior = list(points[:C]) # points inside the fence
    maxy = argmax(interior, key=lambda x: x[1]) # max y index

    corners = [] # list of corners (the thing we want)

    for i in range(C,N):
        p = points[i]
        if p[1] > interior[maxy][1]: continue # outside the fence

        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = argmax(interior, key=lambda x: x[1]) # max y index
        # if (interior[maxy] == p).all(): continue # no corner
        corners.append(np.array((p[0],interior[maxy][1])))


    for c in corners:
        assert count(c) == C
    print(f"{len(points)=}\t{len(corners)=}")
    plt.scatter(points[:,0],points[:,1],label="points")
    corners = np.array(corners)
    plt.scatter(corners[:,0],corners[:,1], marker='x', label="corners")
    plt.legend()
    plt.show()


    ''' 
    This finds all possible points that can be the optimum, by walking
    through each point and finding where it's possible to make the 
    encompassing square, next step is the 3D one
    
    '''

    print("Done!")