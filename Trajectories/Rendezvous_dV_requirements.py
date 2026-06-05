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

MAX_MISSION_TIME = 20 # [years]
LONGP_NUM = 0
AREA_OF_INTEREST = (4,5,17)
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



# === probability functions =====

# def mission_success_probability(dV_budget:int, N:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
#     '''Generates total mission probability for the given scenario.

#     :param dV_budget: Delta V budget of the mission
#     :type dV_budget: int
#     :param N: number of ISOs detected during the mission
#     :type N: int
#     :param rdvz: Whether to consider rendezvous or flyby
#     :type rdvz: bool
#     :param df: dataframe with the ISO data to consider, defaults to None
#     :type df: pd.DataFrame | None, optional
#     :return: success probability
#     :rtype: float
#     '''
#     p_ISO = ISO_probability(dV_budget, rdvz, df)
#     p_least_one = 1-(1-p_ISO)**N
#     return p_least_one

# def ISO_probability(dV_budget:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
#     '''calculate individual chance of success for given dv budget and detection distance,
#     currently only works with integer dV budgets

#     :param dV_budget: Delta V budget of the mission
#     :type dV_budget: int
#     :param rdvz: Whether to consider rendezvous or flyby
#     :type rdvz: bool
#     :param df: dataframe with the ISO data to consider, defaults to None
#     :type df: pd.DataFrame | None, optional
#     :return: probability of intercepting/rendezvousing with one ISO
#     :rtype: float
#     '''
#     # hist = get_dv_hist(rm,weight)
#     # return np.sum(hist[:m.floor(dV_budget)+1])

#     p = dv_below_budget(dV_budget,rdvz, df)
#     return p

# def dv_below_budget(dv_budget:float, rdvz:bool,df:pd.DataFrame|None=None)->float:
#     '''get fraction of orbits that are at or below a dv_budget

#     :param dv_budget: Delta V budget of the mission
#     :type dv_budget: float
#     :param rdvz: Whether to consider rendezvous or flyby
#     :type rdvz: bool
#     :param df: dataframe with the ISO data to consider, defaults to None
#     :type df: pd.DataFrame | None, optional
#     :return: fraction of ISOs in dataframe that are reachable with the given dv budget
#     :rtype: float
#     '''
#     if df is None: df = get_data()
#     if not rdvz:
#         dv = df['icpt_idv']
#     else:
#         dv = df['rdvz_idv'] + df['rdvz_rdv']
#     dv = dv[dv <= dv_budget]
#     return len(dv)/len(df)


def _under(df:pd.DataFrame, dv0:float, dv1:float, dv2:float)->int:
    frac = df[df['h_tdv'] <= dv0]
    frac = frac[frac['h_idv'] <= dv1]
    frac = frac[frac['h_rdv'] <= dv2]
    return len(frac)

def find_best_point(df:pd.DataFrame, N:int):

    count = len(df)
    Pi = 1 - (1-0.9)**(1/N) # needed individual probability
    needed = np.floor(count*Pi) #TODO: change to ceil for more accuracy
    
    # limit to search space:
    
    df = df[df["h_tdv"] <= AREA_OF_INTEREST[0]]
    df = df[df["h_idv"] <= AREA_OF_INTEREST[1]]
    df = df[df["h_rdv"] <= AREA_OF_INTEREST[2]]

    best_row = None
    best_mass = m.inf
    best_count = 0

    for i, row in df.iterrows():
        dv0 = row['h_tdv']
        dv1 = row['h_idv']
        dv2 = row['h_rdv']
        slice = df.loc[
            (df["h_tdv"] <= dv0) &
            (df["h_idv"] <= dv1) &
            (df["h_rdv"] <= dv2)
        ]
        slice_count = len(slice)

        if slice_count > needed:
            if row['h_mass'] < best_mass:
                best_row = row; best_mass = row['h_mass']
            continue
        # else not enough:
        if slice_count >= best_count:
            best_count = slice_count
            if row['h_mass'] < best_mass:
                best_row = row; best_mass = row['h_mass']
            continue
        # else just bad:
        continue
    return best_row


# def mass_view(df:pd.DataFrame, N:int, res:int=20, plot:bool=True):
#     '''plot heatmap of mass for successful schematics'''

    
#     dv0 = np.linspace(0,4,res)
#     dv1 = np.linspace(0,5, res)
#     dv2 = np.linspace(0,15,res)
#     dv0,dv1,dv2 = np.meshgrid(dv0,dv1,dv2)
#     dv0 = dv0.flatten(); dv1 = dv1.flatten(); dv2 = dv2.flatten()
#     mm = []
#     pp = []
#     Pi = 1 - (1-0.9)**(1/N) # needed individual probability

#     for i in tqdm(range(len(dv0)), desc="mass view"):
#         m = 

#         mm.append(m)
#         pp.append(p)
    
#     arg = dv0 > 0
#     dv0 = dv0[arg]
#     dv1 = dv1[arg]
#     dv2 = dv2[arg]
#     mm = np.array(mm)[arg]
#     pp = np.array(pp)[arg]
#     if plot:
#         print("plotting")
#         fig = plt.figure()
#         ax = fig.add_subplot(111,projection='3d')
#         scatter = ax.scatter(dv0,dv1,dv2, c=pp, cmap='PRGn') #type:ignore
#         fig.colorbar(scatter, ax=ax)
#         ax.set_xlabel('turn_dv')
#         ax.set_ylabel('boost_dv')
#         ax.set_zlabel("rendezvous_dv")
#         plt.show()
    
#     idx = np.argmin(mm)
#     print(("rough:" if plot else "fine:"))
#     print("-------")
#     print(f'M: {mm[idx]}')
#     print(f'P: {pp[idx]}')
#     print(f'dv0: {dv0[idx]}')
#     print(f'dv1: {dv1[idx]}')
#     print(f'dv2: {dv2[idx]}')
#     return dv0[idx], dv1[idx], dv2[idx]

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
             "h_turn",'h_rot', "h_idv", "h_rdv", "h_r", "h_t_launch", "h_t_arrival", 'park_longp'
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
        if  not EMERGENCY_SITUATION:
            res = helio_optim(park, ISO, (ISO.tp + MAX_MISSION_TIME*YEAR), detect_t)
        else: res = mad_optim(ISO,(ISO.tp + MAX_MISSION_TIME*YEAR), detect_t, VINF)
        out = ({
            'h_tdv': res['dv0'],
            'h_idv': res['dv1'],
            'h_rdv': res['dv2'],
            'h_mass': res['mass'],
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
            f'h_tdv_{name}' : out1['h_tdv'],
            f'h_idv_{name}' : out1['h_idv'],
            f'h_rdv_{name}' : out1['h_rdv'], 
            f'h_mass_{name}' : out1['h_mass'],
        })
    # make default:
    try:
        out.update(study_ISO(ISO,parking_orbit, detect_t))
    except (ArithmeticError, ValueError, AssertionError): return out
    return out


def study_batch_multi(gen_type:str='', longp_num:int=0)->pd.DataFrame:
    '''multithreaded analysis'''
    
    ISOs = get_ISO(gen_type=gen_type)
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
    print(f"ion delta v cost is: {row["h_tdv"] + row['h_rdv']:.2f} km/s, and relative velocity at intercept is {row['h_rdv']:.2f} km/s, with a boost of {row['h_idv']} km/s, and turn of {row['h_tdv']} km/s")
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
            v = df[f'h_idv_{name}']
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
    # df = get_data(1)
    df = study_batch_multi()
    for i in range(10):
        print(f"batch: {i}")
        df2 = study_batch_multi()
        df = pd.concat((df,df2), ignore_index=True)
    print("interesting fraction:")
    dfi = df[df["h_tdv"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["h_idv"] <= AREA_OF_INTEREST[1]]
    dfi = dfi[dfi["h_rdv"] <= AREA_OF_INTEREST[2]]
    print(dfi[['h_tdv','h_idv','h_rdv','h_mass']])

    print(f" {len(dfi)} / {len(df)} = {len(dfi)/len(df)}")

    point = find_best_point(df, N)
    if point is None: s = 'no valid points'
    else:
        v0 = point['h_tdv']
        v1 = point['h_idv']
        v2 = point['h_rdv']
        m = point['h_mass']
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

        s += (f"best mass: {m:6.0f}  " + ('*' if changed else "") + "kg," +
            f"success chance: {P*100:04.2f}%, " +
            f"delta vees: {v0:04.3f}, {v1:04.3f}, {v2:04.3f} km/s," +
            f"ISOs generated: {len(df):4},"
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

    df = get_data(15)
    print(len(df))
    dfi = df[df['h_tdv'] < 5]
    print(f"cut tdv > 4: {len(dfi)}")
    dfi = dfi[dfi['h_idv'] < 7]
    print(f"cut idv > 7: {len(dfi)}")
    dfi = dfi[dfi['h_rdv']<20]
    print(f"cut rdv > 20: {len(dfi)}")
    plt.hist(df['h_rdv'])
    plt.show()


    # _interp_setup(df)
    print(f"{len(df)=}\t {len(dfi)=}\t frac: {len(dfi)/len(df)}, \n nans: {len(df[np.isnan(df['h_mass'])])}")
    # input()
    dfs = df.sort_values('h_mass', ignore_index=True)
    print(dfs[['h_mass', "h_tdv", "h_idv", "h_rdv", "h_te"]])
    try:
        mass_view(df,350, res=20)
    except ValueError: pass
    print('\n\n')
    dv_optimizer(df, 350)

    # longp_graph(df,prob_needed, LONGP_NUM)

    # plots_for_probability_map() 


    
    
    
    # input()

    run_in_background()
    
    





    # # df = df[df["rdvz_total"] < 19.3]
    # df = df[df['magnitude_generation_method'] == "Omuamua"]
    # # df = df[df['magnitude_generation_method'] == "atlas-borisov"]


    # data = df['detection_r']
    # print(f'detection r: {np.average(data):.3f}')
    # data = df['periapsis']
    # print(f'periapsis: {np.average(data):.3f}')
    # data = df['time_until_periapsis']
    # print(f'time_until_periapsis: {np.average(data):.3f}')
    # print(f'count: {len(df)}')





    # example of using the functions:
    

    # plt.hist(dfb[dfb['rdvz_total'] <= 20]['rdvz_r'],density=True, bins=20)
    # print(f"{len(dfb[dfb['rdvz_total'] <= 20])/len(dfb):.3f}")
    # plt.show()
    # dv_histogram(True,True,dfb)
    # plt.show()
    # probability_map(dfb,True)
    # plt.show()

    
    # dfb = dfb.sort_values('rdvz_total')
    # print(dfb[["rdvz_total", "detection_r","periapsis","time_until_periapsis","rdvz_idv", "rdvz_rdv", "rdvz_r", "rdvz_t_launch", "rdvz_t_arrival"]])



    # run_in_background()
    pass
