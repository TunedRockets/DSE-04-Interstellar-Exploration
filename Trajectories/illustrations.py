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

if True:
    np.random.seed(78091162)
    points = np.random.random((15,2))*0.99
    plt.scatter(points[:,0],points[:,1], marker='x', lw=3, color='b')
    plt.xlabel(r'$\Delta V_i$',fontsize=11)
    plt.ylabel(r'$\Delta V_r$',fontsize=11)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.yticks(ticks=np.arange(0,1.05,0.1),labels='')
    plt.axis()

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
