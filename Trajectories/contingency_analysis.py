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
from Rendezvous_dV_requirements import get_ISO, interpolator_wrapper, _under, Hestia, _argymax
from pathlib import Path
from Structures.holistic_mass_solver import Vesta
from scipy.interpolate import RegularGridInterpolator
import pickle as pkl
import matplotlib.pyplot as plt

MAX_MISSION_TIME = 15 # [years]
MAX_BOOST_DV = 5 # [km/s]
ION_PENALTY = 2
AREA_OF_INTEREST = (20,20) # Area where the mass function is defined


# create simple mass interpolator.
INTERP:RegularGridInterpolator = None # type:ignore
INTERP_PATH = Path(__file__).parent
INTERP_FILE = "Vesta_mass_numbers"

# def _job(p):
#     np.seterr(all="ignore")
#     dv1 = p[0]; dv2 = p[1]
#     try:
#         V = Vesta(0,dv2*1000,dv1*1000)
#         V._converge()
#         return V.lower_stage_wet_mass
#     except (ArithmeticError,ValueError) as e: 
#         return m.nan
    
# def make_interp(res:int = 20):
#     '''interpolator strictly for Vesta model, im km/s'''
#     global INTERP
#     try:
#         with open(INTERP_PATH / (INTERP_FILE + str(res)), 'rb') as file:
#             INTERP = pkl.load(file)
#             return  INTERP;
#     except(FileNotFoundError): pass


#     xx = np.linspace(0,AREA_OF_INTEREST[0]+0.1, res) # boost
#     yy = np.linspace(0,(AREA_OF_INTEREST[1] + AREA_OF_INTEREST[2]) * 3/4 + 0.1, res) # Ion

#     xg,yg = np.meshgrid(xx,yy)
#     xg = xg.flatten(); yg = yg.flatten()
#     pp = np.column_stack((xg,yg))
    
#     # with mp.Pool() as p:
#     #     print("Starting Mass interpolator")
#     #     mm = tqdm(p.imap(_job, pp, 1), desc='Creating interpolator', total=len(pp))
#     # mm = list(mm) # convert to better format

#     mm = []
#     for p in tqdm(pp, desc='single threaded mass calculation'):
#         mm.append(_job(p))

#     mm = np.reshape(np.array(mm),(res,res),)

#     INTERP = RegularGridInterpolator((xx,yy), mm)
#     with open(INTERP_PATH / (INTERP_FILE + str(res)), 'wb') as file:
#         pkl.dump(INTERP,file)
#     return INTERP


def earth_mass(dvi,dvr):
    '''translate dv0:boost, dv1+dv2:ion into the interpolator'''
    # if INTERP is None:
    #     make_interp(); return earth_mass(dv0,dv1,dv2)
    # else:
    #     try: 
    #         return INTERP((dv0,dv1+dv2))
    #     except: return (dv0*5+dv1+dv2)*10_000 + 999_000
    if (dvi > AREA_OF_INTEREST[0] or dvr > AREA_OF_INTEREST[1]): return (dvi*3+dvr)*10_000 + 999_000

    V = Vesta(dvi*1000, dvr*1000, MAX_BOOST_DV, ION_PENALTY)
    V._converge()
    return V.lower_stage_wet_mass # so quick !


    return interpolator_wrapper(0,dv0,dv1+dv2) # Vesta mass interpolator is unreliable...

# def _test_Vesta_mass():

#     for _ in range(200):

#         dv0 = np.random.random()*AREA_OF_INTEREST[0]
#         dv1 = np.random.random()*AREA_OF_INTEREST[1]
#         V = Vesta(0,dv1,dv0)
#         V._converge()
#         vm = V.lower_stage_wet_mass
#         im = earth_mass(dv0,dv1,0)
#         altm = earth_mass(dv1,dv0,0)
#         print(f'accuracy: {(im-vm)/vm:%},\t alternate: accuracy: {(altm-vm)/vm:%}')



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
    
    ISOs = get_ISO(gen_type=gen_type, N_batches=N_batches)
    #for each ISO get row:
    resl = []
    with mp.Pool() as p:
    
        res = tqdm(p.imap_unordered(job, ISOs), desc=f"Studying ISOs, (from earth)", total=len(ISOs))
        resl = list(res)
    return pd.DataFrame(resl)


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
        if (p == pivot).all() and p[1] > interior[maxy][1]:
            return [] #pivot outside, so no new points
        if p[1] > interior[maxy][1]: continue # outside the fence
        interior.pop(maxy) # get rid of highest
        interior.append(p)
        maxy = _argymax(interior) # max y index
        if (i >= pivot_idx): # only add if after the pivot, since otherwise better already exists
            corners.append(np.array((p[0],interior[maxy][1])))

    corners = np.hstack((corners,z*np.ones((len(corners),1))))
    return list(corners)


def find_best_point(df:pd.DataFrame, N:int, P:float=0.9, AOI:tuple[float,float]=AREA_OF_INTEREST, mass_fn=interpolator_wrapper)->tuple[float,float,float,float]:

    count = len(df)
    Pi = 1 - (1-P)**(1/N) # needed individual probability
    needed = m.ceil(count*Pi)
    

    df = df[df["dvi"] <= AOI[0]]
    df = df[df["dvr"] <= AOI[1]]

    ISO_points = df[['dvi', 'dvr']].to_numpy()
    point_list = _bounding_box_solver(ISO_points, needed)
    if len(point_list) == 0: 
        print(f"No point meets probability threshold of {P:.0%}, using best odds possible")
        point_list = [np.array((
        np.max(ISO_points[:,0]),
        np.max(ISO_points[:,1]),
        np.max(ISO_points[:,2]),
        ))]

    best_point = point_list[0]
    best_mass = np.inf
    for point in tqdm(point_list, desc="Finding optimal Point"):
        mass = mass_fn(point[0], point[1], point[2])
        if mass < best_mass:
            best_point = point; best_mass = mass
    
    return best_point[0], best_point[1], best_point[2], best_mass

def direct_earth_analysis(N_batches:int=30):
    '''same as before but now direct from earth'''
    N = 350
    Paim = 0.9 # probability aim

    df = study_batch_earth(N_batches=N_batches)

    print("interesting fraction:")
    dfi = df[df["dvi"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["dvr"] <= AREA_OF_INTEREST[1]]
    print(dfi[['dvi','dvr','mass']])
    count = len(dfi)

    print(f" {count} / {len(df)} = {count/len(df)}")

    point = find_best_point(df, N, Paim, AREA_OF_INTEREST, earth_mass)
    if point is None: s = 'no valid points'
    else:
        v0 = point[0]
        v1 = point[1]
        v2 = point[2]
        m = point[3]
        changed = False
        # get accurate mass:
        if v0 > AREA_OF_INTEREST[0] or v1 > AREA_OF_INTEREST[1] or v2 > AREA_OF_INTEREST[2]:
            try:
                H = Vesta()
                H._converge()
                m = H.lower_stage_wet_mass
                changed = True
            except: m = np.inf


        under = _under(df, v0,v1,v2)
        P = under/len(df)
        P = (1-(1-P)**N)

        s = 'E:'

        s += (f"best mass: {m:>7.0f}" + ('*' if changed else " ") + "kg," +
            f"success chance: {P:>4.2%}, " +
            f"delta vees: ({VINF:3.1f}+) {v0:>6.3f}, {v1:>6.3f} (/{LOW_THRUST_PENALTY:3.1f}), {v2:>6.3f} km/s, " +
            f"ISOs generated: {count:>3}/{len(df):<5}"
        )
    path = Path(__file__).parent / 'runs.txt'
    with open(path, 'a') as file:
        file.write(s + '\n')
        print(s)
    return




if __name__ == '__main__':

    # _test_Vesta_mass()
    # input()
    # H = Hestia(0,18*1000,5*1000, min_acceleration=(7000/(jkat.YEAR)), boost_included_in_acceleration=False)
    # H._converge()
    # print(H)
    # # input()

    V = Vesta(14*1000,10*1000, 3*1000, min_acceleration=(7000/jkat.YEAR), verbose=True, ion_penalty=2)
    V._converge()
    print(V)
    input()

    direct_earth_analysis(100)

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

    print(earth_mass(5,7,7))


    # direct_earth_analysis(20)