import multiprocessing
import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt



def argymax(x:list[np.ndarray]): # argmax for the y coordinate
    idx = 0
    maxx = 0
    for i, p in enumerate(x):
        if p[1] > maxx: maxx = p[1]; idx = i
    return idx


if __name__ == "__main__":

    N = 5000
    C = N // 10
    points = np.random.random((N,2)) # points inside (1,1)

    points = points[np.argsort(points[:,0])] # sort by x
    epsilon = 1e-8
    count = lambda p: len(points[
            (points[:,0] <= p[0]+ epsilon) &
            (points[:,1] <= p[1]+ epsilon)
    ])

    interior = list(points[:C]) # points inside the fence
    maxy = argymax(interior) # max y index

    corners = [] # list of corners (the thing we want)

    # first point is special case:
    pf = points[C-1]
    corners.append(np.array((pf[0],interior[maxy][1])))

    for i in range(C,N):
        p = points[i]
        if p[1] > interior[maxy][1]: continue # outside the fence

        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = argymax(interior) # max y index
        # if (interior[maxy] == p).all(): continue # no corner
        corners.append(np.array((p[0],interior[maxy][1])))


    for c in corners:
        assert count(c) == C
    print(f"{len(points)=}\t{len(corners)=}")
    plt.scatter(points[:,0],points[:,1],label="points")
    corners = np.array(corners)
    if corners.shape == (1,2):
        plt.scatter(corners[0,0],corners[0,1], marker='x', label="corners")
    else:
        plt.scatter(corners[:,0],corners[:,1], marker='x', label="corners")
    plt.legend()
    plt.show()


    ''' 
    This finds all possible points that can be the optimum, by walking
    through each point and finding where it's possible to make the 
    encompassing square, next step is the 3D one
    
    '''

    print("Done!")