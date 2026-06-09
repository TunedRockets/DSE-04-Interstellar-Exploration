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
import os
os.environ["TQDM_DISABLE"] = "1"
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
from functools import partial



MAX_MISSION_TIME = 10 # [years]
MAX_BOOST_DV = 0 # [km/s]
ION_PENALTY = 2
AREA_OF_INTEREST = (16,17) # Area where the mass function is defined


# create simple mass interpolator.
INTERP:RegularGridInterpolator = None # type:ignore
INTERP_PATH = path = Path(__file__).parent.parent / "Structures" / "mass_database_vesta_FH_Exp.pkl"



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

def under2(df:pd.DataFrame, v_inf, v_ion)->int:
    xx = df[['dvi, dvr']].to_numpy()
    xx[:,0] -= v_inf
    yy = np.minimum(0,xx[:,0])*ION_PENALTY + xx[:,1]
    yy = yy[yy<= v_ion]
    return len(yy)

def under3(df:pd.DataFrame, V:Vesta)->int:

    v_inf = V.vinf/1000
    v_ion = V.ion_dv/1000

    xx = df[['dvi', 'dvr']].to_numpy()
    xx = xx - np.column_stack((np.ones(len(xx))*v_inf, np.zeros(len(xx))))
    yy = np.maximum(0,xx[:,0])*ION_PENALTY + xx[:,1]
    yy = yy[yy<= v_ion]
    return len(yy)

def vesta_success_chance(V:Vesta, P:float, N:int, AOI:tuple[float,float]=AREA_OF_INTEREST):
    df = get_data_earth()
    count = len(df)
    C = under3(df,V)

    # remake probability:
    Pi = C/count
    Pu = 1 - (1-Pi)**N
    return Pu




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


def check_ISO_Possible(ISO:jkat.Orbit, detect_t:float,V_inf:float, V_ion:float):


    def residual(dvi,dvr):
        dvi = max(0,(dvi - V_inf))
        dvr += dvi*ION_PENALTY
        return V_ion - dvr

    def study_fn(t):
        ts = t[0]; te = t[1]
        try:
            '''manually for own weighting'''
            r1, v1 = jkat.Earth.t2vectors(ts)
            r2, v2 = ISO.t2vectors(te)
            try: vl1,vl2 = jkat.trajectories.lambert(r1,r2,te-ts,ISO.mu, True)
            except: vl1=vl2=np.array([np.inf,np.inf,np.inf])
            try: va1, va2 = jkat.trajectories.lambert(r1,r2, te-ts, ISO.mu, False)
            except: va1=va2=np.array([np.inf,np.inf,np.inf])
            dvl1 = np.linalg.norm(v1-vl1)
            dvl1 = max(dvl1,0)
            dvl2 = np.linalg.norm(v2-vl2)

            dva1 = np.linalg.norm(v1-va1)
            dva1 = max(dva1,0)
            dva2 = np.linalg.norm(v2-va2)

            # Balance out burns, here dv0 is boost, dv1 is low_thrust as part of initial
            # and dv2 is rendezvous
            
            l = residual(dvl1, dvl2)
            a = residual(dva1,dva2)
            if l >= a:
                return {
                "ts": ts,
                "te": te,
                'dvi': dvl1,
                "dvr": dvl2,
                'r': np.linalg.norm(r2),
                'ion_res': l
            }
            else: return {
                "ts": ts,
                "te": te,
                'dvi': dva1,
                "dvr": dva2,
                'r': np.linalg.norm(r2),
                'ion_res': a
            }
        except: return {'ion_res':-m.inf}

        
    def w(t)->float:
        '''relu for ensuring residual is less'''
        res = study_fn(t)
        return -(res['ion_res'])

    # prescan for minima:
    x0 = prescan_opt(
        w,
        np.linspace(ISO.tp, ISO.tp + 2*jkat.YEAR,10),
        np.linspace(ISO.tp, ISO.tp+MAX_MISSION_TIME*jkat.YEAR, 10)
    )
    res = study_fn(x0)
    if res['ion_res'] >= 0: return res
    # x0 = np.array(((ISO.tp + ISO.tp + jkat.YEAR)/2, (ISO.tp + jkat.YEAR + ISO.tp + 2*jkat.YEAR)/2))
    topt = minimize(w, x0, bounds=((detect_t,detect_t + MAX_MISSION_TIME*jkat.YEAR), (detect_t, detect_t + MAX_MISSION_TIME*jkat.YEAR)))
    if topt.success: return study_fn(topt.x)
    else: return {'ion_res':-m.inf}

def study_batch_possible(V_inf:float, V_ion:float, N_batches:int=30)->pd.DataFrame:
    '''for now bespoke generation, change later'''
    '''multithreaded analysis'''
    print("Getting ISOs...")
    ISOs = get_ISO(N_batches=N_batches)
    #for each ISO get row:
    F = partial(job_possible, V_inf=V_inf, V_ion=V_ion)
    resl = []
    with mp.Pool() as p:
    
        res = tqdm(p.imap_unordered(F, ISOs), desc=f"Studying ISOs, (is possible?)", total=len(ISOs))
        resl = list(res)
    return pd.DataFrame(resl)


def study_storage(V_inf:float, V_ion:float, extra_batches:int=10)->pd.DataFrame:
    '''store for convergence analysis'''
    
    NAME = PICKLE_NAME + f'{V_inf:5.3f}'+ f'{V_ion:5.3f}'

    # generate new if applicable:
    if extra_batches > 0:

        # load my data:
        try:
            data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / (NAME + USER_NAME))
        except (FileNotFoundError):
            data = pd.DataFrame()
        new = [data]

        new.append(study_batch_possible(V_inf,V_ion,extra_batches))
        data = pd.concat(new,ignore_index=True)
        # save result to my data:
        data.to_pickle(PATH_TO_DATA / (NAME + USER_NAME))

    # load all data
    datas = os.listdir(PATH_TO_DATA)
    ldat = []
    for dat in datas:
        if dat.startswith(NAME):
            ldat.append(pd.read_pickle(PATH_TO_DATA / dat))
    mdata = pd.concat(ldat) if len(ldat) > 0 else pd.DataFrame()

    return mdata



    
def job_possible(ISOtuple:tuple[jkat.Orbit, float, str], V_inf:float, V_ion:float)->dict:

    np.seterr(all="ignore")
    ISO, detect_t, g_type = ISOtuple

    detect_r = ISO.r(ISO.f(detect_t))/jkat.AU
    out = {"detection_r":detect_r, "periapsis":ISO.periapsis/jkat.AU, "magnitude_generation_method": g_type,
        'time_until_periapsis':(ISO.tp - detect_t)/jkat.DAY,
            "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.raan, "arg_p":ISO.argp, "t_p":ISO.tp, 
            "ISO_excess_velocity":ISO.vinf}
    try:
        out.update(check_ISO_Possible(ISO, detect_t,V_inf,V_ion))
    except (ArithmeticError, ValueError, AssertionError): return out
    return out


def chance_working(df:pd.DataFrame, N:int=350)->float:

    success = len(df[df['ion_res'] >= 0 ])
    total = len(df)
    Pi = success/total

    Pu = 1 - (1 - Pi)**N
    return Pu


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
    raise NotImplementedError("No longer works")

    df = get_data_earth(extra_batches=N_batches)

    print("interesting fraction:")
    dfi = df[df["dvi"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["dvr"] <= AREA_OF_INTEREST[1]]
    # print(dfi[['dvi','dvr','mass']])
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
                H = Vesta(dvi,dvr,MAX_BOOST_DV,ION_PENALTY, verbose=True)
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




def run_conv(V_inf:float, V_ion:float, N:int):

    chances= []
    while True:
        r = study_storage(V_inf,V_ion)
        c = chance_working(r,N)
        chances.append(c)

        # get std:
        slice_len = 5
        if len(chances) < slice_len:
            sigma = np.nan
        else:
            slice = chances[-slice_len:]
            sigma = np.std(slice)
            if sigma < 1e-3: break
        
        print(f'num_gen: {len(r)},\tprob: {c:%},\t std: {sigma}')


    print("done")



if __name__ == '__main__':


    # direct_earth_analysis(30)

    # # # input()
    # a = 9000 / (jkat.YEAR*2)
    # # V = Vesta(14.7*1000, 8.9*1000, 0, verbose=False, min_acceleration=0, min_engines=2, ion_penalty=2)
    # V = Vesta(11*1000, 8*1000, 0, verbose=False, min_acceleration=a, min_engines=4, ion_penalty=2)
    # V._converge()


    # print(V)
    # input()
    V = Vesta(10*1000,0, verbose=False)
    V._converge()
    print(V)

    df = study_storage(12,9)
    print(f'{chance_working(df ):%}')
    print(f'{len(df[df["ion_res"] >=0])}/{len(df)}')
    plt.hist(df['ion_res'],range=(-100,10),bins=200)
    plt.show()

    run_conv(12,9, 350)

    df = study_batch_possible(12, 9,50)
    print(f'{chance_working(df ):%}')
    plt.hist(df['ion_res'],range=(-100,10),bins=200)
    plt.show()


   

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