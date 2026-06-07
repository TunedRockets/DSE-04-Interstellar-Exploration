'''
Script for generating a distributions of dV expected for the ISO intercept
and from that the mission success chance
missing is a proper distribution of N, which requires more research in the literature
'''

from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import jkat
from jkat import AU, YEAR, DAY
from src.get_ISO import get_ISO, get_cached_ISOs
from src.helio_optim import helio_optim, interpolator_wrapper, mad_optim, get_prob_of_success
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from Structures.holistic_mass_solver import Hestia
# SETTINGS:

PATH_TO_DATA = Path(__file__).parent.parent / "data" 
PICKLE_NAME = "ISOdata"
USER_NAME = os.getlogin()

MAX_MISSION_TIME = 10 # [years]
LONGP_NUM = 0
AREA_OF_INTEREST = (4,7,17) # Area where the mass function is defined
EMERGENCY_SITUATION = False
VINF = 8.5 # + 0.577 
# ap = 5.45 AU
# pe = 10 sun radii
# longp = 124.14 *
# raan = 100.4 *
# i = 1.3 *
from jkat.utils.elements import apse2ae
a,e = apse2ae(5.45*jkat.AU, 10*jkat.SUN_RADIUS)
parking_orbit = jkat.orbit_from_ephemeris(
    a, e, m.radians(1.3), 0, m.radians(200), m.radians(100.4), jkat.SUN_MU
)

def get_parking(longp:float)->jkat.Orbit:
    return jkat.orbit_from_ephemeris(
    a, e, m.radians(1.3), 0, longp, m.radians(100.4), jkat.SUN_MU
)




def _under(df:pd.DataFrame, dv0:float, dv1:float, dv2:float)->int:
    frac = df[df['dv0'] <= dv0]
    frac = frac[frac['dv1'] <= dv1]
    frac = frac[frac['dv2'] <= dv2]
    return len(frac)

def _argymax(x:list[np.ndarray]): # argmax for the y coordinate
    idx = 0
    maxx = 0
    for i, p in enumerate(x):
        if p[1] > maxx: maxx = p[1]; idx = i
    return idx

def _study_slice(points:np.ndarray, pivot:np.ndarray, C:int)->list[np.ndarray]:
    '''study a slice, and add new pivot'''
    if len(points) < C: return [] # no corner here...

    z = pivot[2]
    points = np.vstack((points, pivot)) # add pivot
    points = points[np.argsort(points[:,0])] # sort by x
    interior = list(points[:C]) # points inside the fence
    pivot_idx = np.argwhere((points == pivot)[:,0])[0,0]
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

def _bounding_box_solver(points:np.ndarray, C:int)->list[np.ndarray]:
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
        corners.extend(_study_slice(
            points[:i], points[i], C
        ))
    return corners

def find_best_point(df:pd.DataFrame, N:int, P:float=0.9, AOI:tuple[float,float,float]=AREA_OF_INTEREST, mass_fn=interpolator_wrapper)->tuple[float,float,float,float]:

    count = len(df)
    Pi = 1 - (1-P)**(1/N) # needed individual probability
    needed = m.ceil(count*Pi)
    

    df = df[df["dv0"] <= AOI[0]]
    df = df[df["dv1"] <= AOI[1]]
    df = df[df["dv2"] <= AOI[2]]

    ISO_points = df[['dv0', 'dv1', 'dv2']].to_numpy()
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

# ========== improved storage and study =============
'''
Instead of only storing the end result, store the generated ISO orbit and data about it, that way data can be reanalyzed, and reinterpreted
without generating an entirely new set.

pickle this as a pandas for reuse.
values to store are the 6 orbital parameters gotten from generation (shuffle t_p by a year just in case during generation)
then dv and stats for different types of intercept (flyby, rdvz, jupiter_flyby, jupiter_rdvz)

Units: time in days, speeds in km/s, distances in AU (not applicable to internal values)
'''
col_names = ["detection_r", "periapsis", "magnitude_generation_method", 'time_until_periapsis',
             "parameter", "e", "i", "RAAN", "arg_p", "t_p", 
             "icpt_idv", "icpt_rdv", "icpt_r", "icpt_t_launch", "icpt_t_arrival",
             "rdvz_idv", "rdvz_rdv", "rdvz_r", "rdvz_t_launch", "rdvz_t_arrival",
             "h_turn",'h_rot', "dv1", "dv2", "h_r", "h_t_launch", "h_t_arrival", 'park_longp'
             ]

def study_ISO(ISO:jkat.Orbit, park:jkat.Orbit, detect_t:float)->dict:
    '''study an ISO orbit and return data as a row to be added to a pandas table

    :param ISO: the generated ISO in question
    :type ISO: Orbit
    :param detect_t: time of detection
    :type detect_t: float
    :param gen_type:what type of generation function is used for the absolute magnitude,
    options are 'omuamua' or 'atlas-borisov', if omitted will randomize for each ISO.
    :type gen_type: str
    :return: dict corresponding to pandas row to be added to the database
    :rtype: dict
    '''
    out = {}
    try:

        res = helio_optim(park, ISO, (ISO.tp + MAX_MISSION_TIME*YEAR), detect_t)

        out = ({
            'dv0': res['dv0'],
            'dv1': res['dv1'],
            'dv2': res['dv2'],
            'mass': res['mass'],
            'h_ts' : (res['ts']-detect_t)/DAY,
            'h_te' : (res['te']-detect_t)/DAY,
            'h_r' : res['r']/AU
        })
    except(ArithmeticError, ValueError, AssertionError, KeyError): pass # no intercept :(


    return out

def job(ISOtuple:tuple[jkat.Orbit, float, str], longp_num:int)->dict:

    np.seterr(all="ignore")
    ISO, detect_t, g_type = ISOtuple

    detect_r = ISO.r(ISO.f(detect_t))/AU
    out = {"detection_r":detect_r, "periapsis":ISO.periapsis/AU, "magnitude_generation_method": g_type,
        'time_until_periapsis':(ISO.tp - detect_t)/DAY,
            "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.raan, "arg_p":ISO.argp, "t_p":ISO.tp, 
            "ISO_excess_velocity":ISO.vinf}
    longps = np.linspace(-m.pi, m.pi, longp_num)
    for longp in longps:
        name = f"{m.degrees(longp):3.1f}"
        out1 = study_ISO(ISO,get_parking(longp),detect_t)
        if out1 == {}: continue
        out.update({
            f'dv0_{name}' : out1['dv0'],
            f'dv1_{name}' : out1['dv1'],
            f'dv2_{name}' : out1['dv2'], 
            f'mass_{name}' : out1['mass'],
        })
    # make default:
    try:
        out.update(study_ISO(ISO,parking_orbit, detect_t))
    except (ArithmeticError, ValueError, AssertionError): return out
    return out


def study_batch_multi(gen_type:str='', longp_num:int=0, N_batches:int=20)->pd.DataFrame:
    '''multithreaded analysis'''
    
    ISOs = get_ISO(gen_type=gen_type, N_batches=N_batches)
    F = partial(job, longp_num=longp_num)
    #for each ISO get row:
    resl = []
    with mp.Pool() as p:
    
        res = tqdm(p.imap_unordered(F, ISOs), desc=f"Studying ISOs, (longp_num = {longp_num})", total=len(ISOs))
        resl = list(res)
    return pd.DataFrame(resl)
    



def get_data(extra_batches:int=0, gen_type:str="")->pd.DataFrame:
    '''Get the gathered data on ISOs, 
    also generate a set number of extra batches and add that to the data

    :param extra_batches: number of extra batches to add, defaults to 0
    :type extra_batches: int, optional
    :param gen_type:what type of generation function is used for the absolute magnitude,
    options are 'omuamua' or 'atlas-borisov', if omitted will randomize for each ISO.\n defaults to ''
    :type gen_type: str, optional
    :return: dataframe with the results of the study
    :rtype: pd.DataFrame
    '''
    

    # generate new if applicable:
    if extra_batches > 0:

        # load my data:
        try:
            data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / (PICKLE_NAME + USER_NAME))
        except (FileNotFoundError):
            data = pd.DataFrame()
        new = [data]
        for i in range(extra_batches):
            print('============================================')
            print(f"Generating batch {i+1} of {extra_batches}:")
            print('============================================')
            new.append(study_batch_multi(gen_type, LONGP_NUM))
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

def _fix_data():
    '''Debug function to fix issues with the data'''
    pass
    # =====================

def plot_from_row(row:pd.Series, max_r:float=m.inf):
    '''Plot a 3d representation of the values of a row, plots both rendezvous and intercept trajectories
    get row via: df.iloc[<num>]

    :param ax: matplotlib axes to plot in, needs to be 3d
    :type ax: _type_
    :param row: row to plot
    :type row: pd.Series
    :param max_r: max distance to plot, in AU, if omitted plots up to furthest intercept
    :type max_r: float, optional
    '''

    ISO, t_detect, _ = recreate_ISO(row)


    # recreate parking and helicentric
    res = helio_optim(parking_orbit,ISO, ISO.tp + MAX_MISSION_TIME*YEAR, t_detect)
    ROT:jkat.Orbit = res['ob'] # type: ignore
    HELIO = jkat.orbit_from_lambert(ROT.rvec(0), ISO.t2rvec(res['te']),res['ts'],res['te'], ISO.mu)




    # plot earth, jupiter and iso:
    jkat.plot(ISO,t_bounds=(t_detect, res['te']), max_distance=max_r, label="ISO", color='pink')
    jkat.add_solar_system()

    # get the parking orbit:
    jkat.plot(parking_orbit,label="parking orbit", color='gray')
    jkat.plot(ROT,label="turned parking orbit", color='lightgray')
    jkat.plot(HELIO, t_bounds=(res['ts'], res['te']), label="Heliocentric intercept", max_distance=max_r, color="purple")

    # printing:

    print(f'Helio:\nlaunches: {row["h_ts"]:.2f} days after detection, arrives {row["h_te"]:.2f} days after detection at a distance of {row["h_r"]} AU')
    print(f"ion delta v cost is: {row["dv0"] + row['dv2']:.2f} km/s, and relative velocity at intercept is {row['dv2']:.2f} km/s, with a boost of {row['dv1']} km/s, and turn of {row['dv0']} km/s")
    plt.legend()
    plt.show()

def recreate_ISO(row:pd.Series)->tuple[jkat.Orbit,float,str]:
    '''recreate the ISO orbit, detection time, and gen_type from row

    :param row: _description_
    :type row: pd.Series
    :return: ISO orbit
    :return: ISO detection time
    :return: ISO generation method
    :rtype: tuple[Orbit,float,str]
    '''
    # extract orbit:
    ISO = jkat.Orbit(
        row['parameter'],
        row['e'],
        row['i'],
        row['RAAN'],
        row['arg_p'],
        row['t_p'],
        jkat.SUN_MU
    )
    detect_r = row['detection_r']
    t_detect = ISO.t(-ISO.cross_radius(detect_r*AU)) # type:ignore
    return ISO, t_detect, row['magnitude_generation_method']

def longp_graph(df:pd.DataFrame, fraction:float, longp_num:int = 0):

    pp = []
    vv = []
    ww = []

    for longp in np.linspace(-m.pi,m.pi, longp_num):
        pp.append(longp)
        name = f"{m.degrees(longp):3.1f}"
        try:
            v = df[f'dv1_{name}']
            v = v.sort_values(ignore_index=True)
            n = m.ceil(len(v)*fraction)
            vreq = v[n]
            wreq = v[n-1]
            vv.append(vreq)
            ww.append(wreq)
            print(f" for {name}: number required is: {n} out of {len(v)}, corresponding to: {wreq:3.2f} --- {vreq:3.2f} km/s")
        except:
            vv.append(m.nan)
            ww.append(m.nan)
            print("missing column")
    plt.polar(pp,vv, label='upper bound')
    plt.polar(pp,ww, label='lower bound')
    plt.title('oberth dv required per longitude of periapsis')
    plt.legend()
    plt.show()
        
def run_in_background():
    '''run forever generating new datapoints'''
    while True: 
        df = get_data(1)
        print("Current # of rows:")
        print(len(df))
        print('---------\n')


def we_am_going_insane():
    '''Crazy? I was crazy once. They locked me in a room. A rubber room. A rubber room with rats, and rats make me crazy. Crazy? I was crazy once. They locked me in a room. A rubber room. A rubber room with rats, and rats make me crazy. Crazy? I was crazy once. They locked me in a room. A rubber room. A rubber room with rats, and rats make me crazy'''
    N = 350
    Paim = 0.9 # probability aim
    # df = get_data(1)
    df = study_batch_multi(N_batches=30)
    # for i in range(10):
    #     print(f"batch: {i}")
    #     df2 = study_batch_multi()
    #     df = pd.concat((df,df2), ignore_index=True)
    print("interesting fraction:")
    dfi = df[df["dv0"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["dv1"] <= AREA_OF_INTEREST[1]]
    dfi = dfi[dfi["dv2"] <= AREA_OF_INTEREST[2]]
    print(dfi[['dv0','dv1','dv2','mass']])
    count = len(dfi)

    print(f" {count} / {len(df)} = {count/len(df)}")

    point = find_best_point(df, N, Paim) # CHANGE THE ODDS
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
                H = Hestia(v0*1000, v2*1000, v1*1000, False,0.001)
                H._converge()
                m = H.lower_stage_wet_mass
                changed = True
            except: m = np.inf


        under = _under(df, v0,v1,v2)
        P = under/len(df)
        P = (1-(1-P)**N)

        s = 'D:' if EMERGENCY_SITUATION else ''

        s += (f"best mass: {m:>7.0f}" + ('*' if changed else " ") + "kg," +
            f"success chance: {P:4.2%}, " +
            f"delta vees: {v0:06.3f}, {v1:06.3f}, {v2:06.3f} km/s, " +
            f"ISOs generated: {count:>3}/{len(df):<5}"
        )
    path = Path(__file__).parent / 'runs.txt'
    with open(path, 'a') as file:
        file.write(s + '\n')
        print(s)
    return




# ======= plotting and analysis ========

if __name__ == "__main__":



    while True:
        # get_cached_ISOs(1)
        we_am_going_insane()
    # run_in_background()  
    pass
