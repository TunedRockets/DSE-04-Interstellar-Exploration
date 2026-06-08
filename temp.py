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

def study_slice(points:np.ndarray, pivot:np.ndarray, C:int)->list[np.ndarray]:
    '''study a slice, and add new pivot'''

    if len(points) < C: return [] # no corner here...

    z = pivot[2]
    points = np.vstack((points, pivot)) # add pivot
    points = points[np.argsort(points[:,0])] # sort by x

    interior = list(points[:C]) # points inside the fence
    pivot_idx = np.argwhere((points == pivot)[:,0])[0,0]
    maxy = argymax(interior) # max y index
    corners = [] # list of corners (the thing we want)

    # first point is special case:
    pf = points[C-1]
    corners.append(np.array((pf[0],interior[maxy][1])))

    for i in range(C,len(points)):
        p = points[i]

        # pivot check
        ...
        if (p == pivot).all() and p[1] > interior[maxy][1]:
            return [] #pivot outside, so no 


        if p[1] > interior[maxy][1]: continue # outside the fence
        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = argymax(interior) # max y index

        if (i >= pivot_idx): # only add if after the pivot, since otherwise better already exists
            corners.append(np.array((p[0],interior[maxy][1])))
    corners = np.hstack((corners,z*np.ones((len(corners),1))))
    return list(corners)


def minimum_bounding_boxes(points:np.ndarray, C:int)->list[np.ndarray]:
    '''find all coordinates that cover C points
    i think it grows by N^2, it can run quite quick with N<200 and even N<500.
    so i think it's good enough
    '''

    points = points[np.argsort(points[:,2])] # sort by z
    if len(points) < C: return []
    elif len(points) == C: return [np.array(
        (np.max(points[:,0]),np.max(points[:,1]),np.max(points[:,2]))
    )]
    corners = []
    for i in tqdm(range(C-1,len(points)), desc="finding bounding boxes"):
        corners.extend(study_slice(
            points[:i], points[i], C
        ))
    return corners




if __name__ == "__main__":

    N = 50
    C = N//10
    points = np.random.random((N,3)) # points inside (1,1)

    count = lambda p: len(points[
            (points[:,0] <= p[0]) &
            (points[:,1] <= p[1]) &
            (points[:,2] <= p[2])
    ])
    corners = minimum_bounding_boxes(points, C)
   
    for c in corners:
        assert count(c) == C
    print(f"{len(points)=}\t{len(corners)=}")
    ax = plt.figure().add_subplot(projection='3d')


    ax.scatter(points[:,0],points[:,1],points[:,2],label="points") # type:ignore
    corners = np.array(corners)
    if corners.shape == (1,3):
        ax.scatter(corners[0,0],corners[0,1],corners[0,2], marker='x', label="corners")
    else:
        ax.scatter(corners[:,0],corners[:,1], corners[:,2], marker='x', label="corners") # type:ignore
    plt.legend()
    plt.show()


    ''' 
    This finds all possible points that can be the optimum, by walking
    through each point and finding where it's possible to make the 
    encompassing square, next step is the 3D one
    
    '''

    print("Done!")