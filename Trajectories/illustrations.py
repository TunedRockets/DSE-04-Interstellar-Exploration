''' 
a picture says a thousand words, so this way we save on page count
'''



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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


# drawing random orbits
if False:

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
   

# cmap of success chance
if True:
    # (pessemistic since trajectories not reoptimized)
    def under(df:pd.DataFrame,vinf, dvion):
        count = 0
        for i, row in df.iterrows():
            dvi = row['dvi']
            dvr = row['dvr']
            dvi = max(0, dvi- vinf)
            if 2*dvi + dvr <= dvion: count += 1
        return count

    def pct_chance(df:pd.DataFrame, count:int, vinf:float, dvion: float, N:int):

        frac = under(df,vinf,dvion)/count
        chance = 1 - (1-frac)**N
        return chance
   
        
    from contingency_analysis import study_storage

    df = study_storage(12,10,0)
    lendf = len(df)
    df = df[df['ion_res'] >= 0]

    # TODO: do this for meshgrid then plot
    # and redo _under to use Vinf and Dvion instead...
    res = 100
    vvion = np.linspace(6,10.5,res)
    vvinf = np.linspace(10,12.5,res)
    PP = []
    for vinf in tqdm(vvinf, desc='making plot'):
        Prow = []
        for vion in vvion:
            Prow.append(pct_chance(df,lendf,vinf,vion,350))
        PP.append(Prow)
    PP = np.array(PP).T

    plt.imshow(PP, origin='lower', aspect=1/2,
               extent=(vvinf[0],vvinf[-1],vvion[0],vvion[-1]))
    plt.colorbar(location="right", label=r"$P_s$")
    
    plt.scatter(12,10, color="red")
    CS = plt.contour(PP,levels=[0.5, 0.75,0.9,0.95],origin="lower", extent=(vvinf[0],vvinf[-1],vvion[0],vvion[-1]), colors='k')
    plt.clabel(CS, fmt=lambda x: f"{x:.0%}")
    # plt.axis('scaled')
    plt.xlabel(r'$V_\infty$')
    plt.ylabel(r'$\Delta V_{\rm ion}$')

    

    plt.show()




    
    # N_range = np.arange(10,MS_N + 30,5)
    # V_range =np.arange(1,25, 0.5)
    # NN, VV = np.meshgrid(N_range,V_range)
    # F = lambda v,n: mission_success_probability(v,n,rdvz,df)
    # PP = np.vectorize(F)(VV,NN)
    # plt.imshow(PP,origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    # if num != 1:
    #     plt.colorbar(location="right", label=r"$P_s$")
    # CS = plt.contour(PP,levels=[0.5,0.9,0.99],origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]), colors='k')
    # plt.clabel(CS, fmt=lambda x: f"{x*100:.0f}%")
    # if num < 2:
    #     plt.ylabel(r'$\Delta V$ budget [km/s]')
    # plt.xlabel(r'$N$')
    # # plt.title(f"Probability map for {"rendezvous" if rdvz else "intercept"}\nAnd estimated ISO detections during {years} year mission")
    # if guesses:
    #     plt.axvline(EL_N,ls='--', color="gray")
    #     plt.text(EL_N+1, np.average(V_range)+3, "Ezell, Loeb mean", color="gray")
    #     plt.axvline(HSP_N,ls='--', color="gray")
    #     plt.text(HSP_N+1, np.average(V_range), "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
    #     plt.axvline(MS_N,ls='--', color="gray")
    #     plt.text(MS_N-1, np.average(V_range)-3, "Marčeta, Seligman mean", ha="right", color="gray")

    # plt.gca().set_aspect(N_range[-1]/V_range[-1])
