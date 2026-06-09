'''
Using a direct from earth trajectory, with a comparable 
Vinf impluse as would be required for the heliocentric case.
Uses a dV penalty for the Ion stage to simulate low-thrust trajectory
(since the majority of the burn, up to escape velocity speed, is impulsive,
this should be broadly accurate).
here, dv0 is the boost stage dv, dv1 is the extra required by the ions,
and dv2 is the rendezvous burn.

running this shows that earth is definitely feasible. 


TODO: need better heat shield sizer!!! (check with Sem)


'''
import jkat
import pandas as pd
import numpy as np
import math as m
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import minimize
from Rendezvous_dV_requirements import get_ISO, Hestia, _argymax
from Structures.Vesta_interpolation import MassInterpolator
from pathlib import Path
from Structures.holistic_mass_solver import Vesta
from scipy.interpolate import RegularGridInterpolator
import pickle as pkl
import matplotlib.pyplot as plt
import os

MAX_MISSION_TIME = 10 # [years]
MAX_BOOST_DV = 0 # [km/s]
ION_PENALTY = 2
AREA_OF_INTEREST = (20,20) # Area where the mass function is defined


# create simple mass interpolator.
INTERP:RegularGridInterpolator = None # type:ignore
INTERP_PATH = path = Path(__file__).parent.parent / "Structures" / "mass_database_vesta.pkl"



def earth_mass(dvi,dvr):
    '''translate dv0:boost, dv1+dv2:ion into the interpolator'''
    global INTERP
    if INTERP is None:
        INTERP = MassInterpolator(INTERP_PATH).interp 
    try:
        return INTERP((1000*dvi, 1000*dvr))
    except ValueError: return (dvi*2000 + dvr*1000)*50 + 99_999


    V = Vesta(dvi*1000, dvr*1000, MAX_BOOST_DV, ION_PENALTY)
    V._converge()
    return V.lower_stage_wet_mass # so quick !


def _under(df:pd.DataFrame, dvi:float, dvr:float)->int:
    frac = df[df['dvi'] <= dvi]
    frac = frac[frac['dvr'] <= dvr]
    return len(frac)



def study_ISO(ISO:jkat.Orbit, detect_t:float)->dict:
    '''study earth based intercept'''
    def F(t):
        try:
            '''manually for own weighting'''
            t1 = t[0]; t2 = t[1]
            r1, v1 = jkat.Earth.t2vectors(t1)
            r2, v2 = ISO.t2vectors(t2)
            try: vl1,vl2 = jkat.trajectories.lambert(r1,r2,t2-t1,ISO.mu, True)
            except: vl1=vl2=np.array([np.inf,np.inf,np.inf])
            try: va1, va2 = jkat.trajectories.lambert(r1,r2, t2-t1, ISO.mu, False)
            except: va1=va2=np.array([np.inf,np.inf,np.inf])
            dvl1 = np.linalg.norm(v1-vl1)
            dvl1 = max(dvl1,0)
            dvl2 = np.linalg.norm(v2-vl2)

            dva1 = np.linalg.norm(v1-va1)
            dva1 = max(dva1,0)
            dva2 = np.linalg.norm(v2-va2)

            # Balance out burns, here dv0 is boost, dv1 is low_thrust as part of initial
            # and dv2 is rendezvous
            
            lmass = earth_mass(dvl1, dvl2)
            amass = earth_mass(dva1,dva2)
            if lmass < amass:
                return {
                "ts": t1,
                "te": t2,
                'dvi': dvl1,
                "dvr": dvl2,
                'r': np.linalg.norm(r2),
                'mass': lmass
            }
            else: return {
                "ts": t1,
                "te": t2,
                'dvi': dva1,
                "dvr": dva2,
                'r': np.linalg.norm(r2),
                'mass': amass
            }
        except: return {'dvi':m.inf, 'dvr': m.inf, 'mass': m.inf}
    
    def w(t): 
        try: return F(t)['mass']
        except(ValueError, ArithmeticError): return m.inf


    # prescan for minima:
    x0 = prescan_opt(
        w,
        np.linspace(ISO.tp, ISO.tp + 2*jkat.YEAR,10),
        np.linspace(ISO.tp, ISO.tp+MAX_MISSION_TIME*jkat.YEAR, 10)
    )
    # x0 = np.array(((ISO.tp + ISO.tp + jkat.YEAR)/2, (ISO.tp + jkat.YEAR + ISO.tp + 2*jkat.YEAR)/2))
    topt = minimize(w, x0, bounds=((detect_t,detect_t + MAX_MISSION_TIME*jkat.YEAR), (detect_t, detect_t + MAX_MISSION_TIME*jkat.YEAR)))
    if topt.success: return F(topt.x)
    else: return {}

def prescan_opt(F, xx, yy):
    '''prescan for a sorta global minima of the function'''

    xg, yg = np.meshgrid(xx,yy)
    xg = xg.flatten(); yg = yg.flatten();
    ww = []
    for i in range(len(xg)):
        ww.append(F((xg[i],yg[i])))
    idx = np.array(ww).argmin()
    return (xg[idx],yg[idx])



def job(ISOtuple:tuple[jkat.Orbit, float, str])->dict:

    np.seterr(all="ignore")
    ISO, detect_t, g_type = ISOtuple

    detect_r = ISO.r(ISO.f(detect_t))/jkat.AU
    out = {"detection_r":detect_r, "periapsis":ISO.periapsis/jkat.AU, "magnitude_generation_method": g_type,
        'time_until_periapsis':(ISO.tp - detect_t)/jkat.DAY,
            "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.raan, "arg_p":ISO.argp, "t_p":ISO.tp, 
            "ISO_excess_velocity":ISO.vinf}
    try:
        out.update(study_ISO(ISO, detect_t))
    except (ArithmeticError, ValueError, AssertionError): return out
    return out

def study_batch_earth(gen_type:str='', N_batches:int=20)->pd.DataFrame:
    '''multithreaded analysis'''
    print("Getting ISOs...")
    ISOs = get_ISO(gen_type=gen_type, N_batches=N_batches)
    #for each ISO get row:
    resl = []
    with mp.Pool() as p:
    
        res = tqdm(p.imap_unordered(job, ISOs), desc=f"Studying ISOs, (from earth)", total=len(ISOs))
        resl = list(res)
    return pd.DataFrame(resl)



PATH_TO_DATA = Path(__file__).parent.parent / "data" 
PICKLE_NAME = "EISOdata"
USER_NAME = os.getlogin()

def get_data_earth(extra_batches:int=0, gen_type:str="")->pd.DataFrame:
    '''Get the gathered data on ISOs, 
    using the direct earth method. generated fields are:
    "detection_r","periapsis","magnitude_generation_method",'time_until_periapsis',"parameter",
    "e", "i", "RAAN", "arg_p", "t_p", "ISO_excess_velocity"

    as well as:
    "ts", "te", 'dvi', "dvr", 'r', 'mass'
    '''

    # generate new if applicable:
    if extra_batches > 0:
        # load my data:
        try:
            data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / (PICKLE_NAME + USER_NAME))
        except (FileNotFoundError):
            data = pd.DataFrame()
        new = [data]
        new.append(study_batch_earth(gen_type, extra_batches))
        data = pd.concat(new,ignore_index=True)
        # save result to my data:
        data.to_pickle(PATH_TO_DATA / (PICKLE_NAME + USER_NAME))

    # load all data
    datas = os.listdir(PATH_TO_DATA)
    ldat = []
    for dat in datas:
        if dat.startswith(PICKLE_NAME):
            ldat.append(pd.read_pickle(PATH_TO_DATA / dat))
    mdata = pd.concat(ldat) if len(ldat) > 0 else pd.DataFrame()

    return mdata


def bounding_box_2d(points, C):

    if len(points) < C: return [] # no corner here...

    points = points[np.argsort(points[:,0])] # sort by x
    interior = list(points[:C]) # points inside the fence
    maxy = _argymax(interior) # max y index
    corners = [] # list of corners (the thing we want)

    # first point is special case:
    pf = points[C-1]
    corners.append(np.array((pf[0],interior[maxy][1])))

    for i in range(C,len(points)):
        p = points[i]
        if p[1] > interior[maxy][1]: continue # outside the fence
        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = _argymax(interior) # max y index
        corners.append(np.array((p[0],interior[maxy][1])))

    return list(corners)


def find_best_point(df:pd.DataFrame, N:int, P:float=0.9, AOI:tuple[float,float]=AREA_OF_INTEREST)->tuple[float,float,float]:

    count = len(df)
    Pi = 1 - (1-P)**(1/N) # needed individual probability
    needed = m.ceil(count*Pi)
    

    df = df[df["dvi"] <= AOI[0]]
    df = df[df["dvr"] <= AOI[1]]

    ISO_points = df[['dvi', 'dvr']].to_numpy()
    point_list = bounding_box_2d(ISO_points, needed)
    if len(point_list) == 0: 
        print(f"No point meets probability threshold of {P:.0%}, using best odds possible")
        point_list = [np.array((
        np.max(ISO_points[:,0]),
        np.max(ISO_points[:,1])
        ))]

    best_point = point_list[0]
    best_mass = np.inf
    for point in tqdm(point_list, desc="Finding optimal Point"):
        mass = earth_mass(point[0], point[1])
        if mass < best_mass:
            best_point = point; best_mass = mass
    
    return best_point[0], best_point[1], best_mass

def direct_earth_analysis(N_batches:int=30):
    '''same as before but now direct from earth'''
    N = 350
    Paim = 0.9 # probability aim

    df = get_data_earth(extra_batches=N_batches)

    print("interesting fraction:")
    dfi = df[df["dvi"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["dvr"] <= AREA_OF_INTEREST[1]]
    print(dfi[['dvi','dvr','mass']])
    count = len(dfi)

    print(f" {count} / {len(df)} = {count/len(df)}")

    point = find_best_point(df, N, Paim, AREA_OF_INTEREST)
    if point is None: s = 'no valid points'
    else:
        dvi = point[0]
        dvr = point[1]

        m = point[2]
        changed = False
        # get accurate mass:
        if dvi > AREA_OF_INTEREST[0] or dvr > AREA_OF_INTEREST[1]:
            try:
                H = Vesta(dvi,dvr,MAX_BOOST_DV,ION_PENALTY)
                H._converge()
                m = H.lower_stage_wet_mass
                changed = True
            except: m = np.inf


        under = _under(df, dvi,dvr)
        P = under/len(df)
        P = (1-(1-P)**N)

        s = 'E:'

        s += (f"best mass: {m:>7.0f}" + ('*' if changed else " ") + "kg," +
            f"success chance: {P:>4.2%}, " +
            f"delta vees: {dvi:5.3f}, {dvr:5.3f} km/s, " +
            f"ISOs generated: {count:>3}/{len(df):<5}"
        )
    path = Path(__file__).parent / 'runs.txt'
    with open(path, 'a') as file:
        file.write(s + '\n')
        print(s)
    return dvi, dvr




if __name__ == '__main__':


    # print(earth_mass(5,7))


    # input()
    # a = 9000 / (jkat.YEAR)
    # V = Vesta(14.73*1000, 4.153*1000, 0, verbose=True, min_acceleration=a)
    # V._converge()
    # print(V)
    # input()

    dvi, dvr = direct_earth_analysis(20)
    V = Vesta(dvi*1000, dvr*1000, 5000, verbose=False)
    V._converge()
    print(V)

    # i = make_interp(20)

    # xx = np.linspace(0,5)
    # yy = np.linspace(0,20)
    # pp = []
    # for y in yy:
    #     prow = []
    #     for x in xx:
    #         prow.append(i((x,y)))
    #     pp.append(prow)

    # plt.imshow(pp, origin='lower', extent=(0,5,0,20))
    # plt.axis('scaled')
    # plt.show()




    # direct_earth_analysis(20)