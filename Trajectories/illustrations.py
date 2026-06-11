''' 
a picture says a thousand words, so this way we save on page count
'''



import matplotlib.pyplot as plt
import numpy as np


# fraction showcase:
if False:
    np.random.seed(122)
    points = np.random.random((20,2))*0.99
    plt.scatter(points[:,0],points[:,1], marker='x', lw=3, color='b')
    plt.xlabel(r'$\Delta V_i$',fontsize=11)
    plt.ylabel(r'$\Delta V_r$',fontsize=11)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.yticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.axis()

    design_point = np.array((0.22,0.72))
    count = len(points[
        (points[:,0] <= design_point[0]) &
        (points[:,1] <= design_point[1]) 
    ])
    print(f"probability: {count}/{len(points)}")
    plt.plot([design_point[0],design_point[0]],[0,design_point[1]], color='k', lw=1, zorder=-99)
    plt.plot([0,design_point[0]],[design_point[1],design_point[1]], color='k', lw=1, zorder=-99)
    plt.scatter(design_point[0],design_point[1], color="green", marker='s', lw=3, label=f"design point with probability $P={count}/{20}$")

    design_point = np.array((0.35,0.3))
    count = len(points[
        (points[:,0] <= design_point[0]) &
        (points[:,1] <= design_point[1]) 
    ])
    print(f"probability: {count}/{len(points)}")
    plt.plot([design_point[0],design_point[0]],[0,design_point[1]], color='k', lw=1, zorder=-99)
    plt.plot([0,design_point[0]],[design_point[1],design_point[1]], color='k', lw=1, zorder=-99)
    plt.scatter(design_point[0],design_point[1], color="red", marker='o', lw=3, label=f"design point with probability $P={count}/{20}$")

    plt.legend()
    plt.show()


# specific design optimizer:
if False:
    np.random.seed(121)
    points = np.random.random((10,2))*0.99
    plt.scatter(points[:,0],points[:,1], marker='x', lw=3, color='b')
    plt.xlabel(r'$\Delta V_i$',fontsize=11)
    plt.ylabel(r'$\Delta V_r$',fontsize=11)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.yticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.axis()

    design_point = np.array((0.5,0.45))
    count = len(points[
        (points[:,0] <= design_point[0]) &
        (points[:,1] <= design_point[1]) 
    ])
    print(f"probability: {count}/{len(points)}")
    plt.plot([design_point[0],design_point[0]],[0,design_point[1]], color='k', lw=1, zorder=-99)
    plt.plot([0,design_point[0]],[design_point[1],design_point[1]], color='k', lw=1, zorder=-99)
    plt.scatter(design_point[0],design_point[1], color="green", marker='s', lw=3)

    # moved points:
    move = np.array((
        (0.1,0.6),
        (0.3,0.42),
        ))
    plt.plot(move[:,0], move[:,1], color="b", ls='--', lw=1)
    plt.scatter(move[-1,0], move[-1,1], color='b', marker='x', lw=3)
     # moved points:
    move = np.array((
        (0.6,0.2),
        (0.48,0.38),
        ))
    plt.plot(move[:,0], move[:,1], color="b", ls='--', lw=1)
    plt.scatter(move[-1,0], move[-1,1], color='b', marker='x', lw=3)

    move = np.array((
        (0.55,0.07),
        (0.48,0.23),
        ))
    plt.plot(move[:,0], move[:,1], color="b", ls='--', lw=1)
    plt.scatter(move[-1,0], move[-1,1], color='b', marker='x', lw=3)


    # plt.legend()
    plt.show()

if False:
    np.random.seed(78091162)
    points = np.random.random((15,2))*0.99
    plt.scatter(points[:,0],points[:,1], marker='x', lw=3, color='b')
    plt.xlabel(r'$\Delta V_i$',fontsize=11)
    plt.ylabel(r'$\Delta V_r$',fontsize=11)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.yticks(ticks=np.arange(0,1.05,0.1),labels='')

    design_point = np.array((0.5,0.6))
    count = len(points[
        (points[:,0] <= design_point[0]) &
        (points[:,1] <= design_point[1]) 
    ])
    print(f"probability: {count}/{len(points)}")
    plt.plot([design_point[0] + design_point[1]/2,design_point[0]],[0,design_point[1]], color='k', lw=1, zorder=-99)
    plt.plot([design_point[0],design_point[0]],[0,design_point[1]], color='k', lw=1, ls='--', zorder=-99)
    

    plt.plot([0,design_point[0]],[design_point[1],design_point[1]], color='k', lw=1, zorder=-99)
    plt.scatter(design_point[0],design_point[1], color="green", marker='s', lw=3)

    plt.show()


# optimum search space

def argymax(x:list[np.ndarray]): # argmax for the y coordinate
    idx = 0
    maxx = 0
    for i, p in enumerate(x):
        if p[1] > maxx: maxx = p[1]; idx = i
    return idx


def study_slice(points:np.ndarray, C:int)->np.ndarray:
    '''study a slice, and add new pivot'''

    if len(points) < C: return np.array([]) # no corner here...



    points = points[np.argsort(points[:,0])] # sort by x

    interior = list(points[:C]) # points inside the fence
    maxy = argymax(interior) # max y index
    corners = [] # list of corners (the thing we want)

    # first point is special case:
    pf = points[C-1]
    corners.append(np.array((pf[0],interior[maxy][1])))

    for i in range(C,len(points)):
        p = points[i]

        if p[1] > interior[maxy][1]: continue # outside the fence
        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = argymax(interior) # max y index

        corners.append(np.array((p[0],interior[maxy][1])))
    return np.array(corners)

# optimization stairs
if False:
    N = 30
    C = 5
    points = np.random.random((N,2))*0.97 # points inside (1,1)
    print(f'{N=},{C=}')

    corners = study_slice(points, C)

    border = [np.array((corners[0,0], 1))]

    for i in range(len(corners)-1):
        border.append(corners[i])
        border.append(np.array([
            corners[i+1][0], corners[i][1]
        ]))
    border.append(corners[-1])
    border.append(np.array([
        1, corners[-1][1]
    ]))

    border = np.array(border)
    plt.plot(border[:,0],border[:,1], ls='--', lw=1, zorder=-99, color='orange')
        



    plt.scatter(points[:,0],points[:,1],label="ISOs", marker='x', color='b', lw=3)

    plt.scatter(corners[:,0],corners[:,1], color='orange', label="potential design points")

    plt.xlabel(r'$\Delta V_i$',fontsize=11)
    plt.ylabel(r'$\Delta V_r$',fontsize=11)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.yticks(ticks=np.arange(0,1.05,0.1),labels='')

    plt.legend()
    plt.show()



if True:

    from contingency_analysis import study_storage, recreate_ISO_and_intercept 
    import jkat
    from jkat.plotting.plot import init

    df = study_storage(12,10,0)
    df = df[df['ion_res'] >= 0]
    df = df.sort_values('r', ignore_index=True)

    row = df.iloc[-2]
    td = row['t_p']- row['time_until_periapsis']*jkat.DAY

    ISO, trans, ts, te = recreate_ISO_and_intercept(row)


    jkat.add_solar_system(ts, '11111111', True)

    jkat.plot(ISO, t_bounds=(td,te), t=ts, max_distance=50*jkat.AU, color="purple")
    jkat.plot(trans,t_bounds=(ts,te), t=te, color='green', stilts=True)
    jkat.show()
   