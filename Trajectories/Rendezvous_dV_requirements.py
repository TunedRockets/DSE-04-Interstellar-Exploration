'''
Script for generating a distributions of dV expected for the ISO intercept
and from that the mission success chance
missing is a proper distribution of N, which requires more research in the literature
'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import jkat
from jkat import AU, YEAR, DAY
from src.get_ISO import get_ISO
from src.helio_optim import helio_optim
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
import pandas as pd

# SETTINGS:

PATH_TO_DATA = Path(__file__).parent.parent / "data" 
PICKLE_NAME = "ISOdata.pic"

MAX_MISSION_TIME = 10 # [years]
MAX_BOOST_DV = 4
LONGP_NUM = 12


# ap = 5.45 AU
# pe = 10 sun radii
# longp = 124.14 *
# raan = 100.4 *
# i = 1.3 *
from jkat.utils.elements import apse2ae
a,e = apse2ae(5.45*jkat.AU, 10*jkat.SUN_RADIUS)
parking_orbit = jkat.orbit_from_ephemeris(
    a, e, m.radians(1.3), 0, m.radians(124.14), m.radians(100.4), jkat.SUN_MU
)

def get_parking(longp:float)->jkat.Orbit:
    return jkat.orbit_from_ephemeris(
    a, e, m.radians(1.3), 0, longp, m.radians(100.4), jkat.SUN_MU
)



# === probability functions =====

def mission_success_probability(dV_budget:int, N:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
    '''Generates total mission probability for the given scenario.

    :param dV_budget: Delta V budget of the mission
    :type dV_budget: int
    :param N: number of ISOs detected during the mission
    :type N: int
    :param rdvz: Whether to consider rendezvous or flyby
    :type rdvz: bool
    :param df: dataframe with the ISO data to consider, defaults to None
    :type df: pd.DataFrame | None, optional
    :return: success probability
    :rtype: float
    '''
    p_ISO = ISO_probability(dV_budget, rdvz, df)
    p_least_one = 1-(1-p_ISO)**N
    return p_least_one

def ISO_probability(dV_budget:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
    '''calculate individual chance of success for given dv budget and detection distance,
    currently only works with integer dV budgets

    :param dV_budget: Delta V budget of the mission
    :type dV_budget: int
    :param rdvz: Whether to consider rendezvous or flyby
    :type rdvz: bool
    :param df: dataframe with the ISO data to consider, defaults to None
    :type df: pd.DataFrame | None, optional
    :return: probability of intercepting/rendezvousing with one ISO
    :rtype: float
    '''
    # hist = get_dv_hist(rm,weight)
    # return np.sum(hist[:m.floor(dV_budget)+1])

    p = dv_below_budget(dV_budget,rdvz, df)
    return p

def dv_below_budget(dv_budget:float, rdvz:bool,df:pd.DataFrame|None=None)->float:
    '''get fraction of orbits that are at or below a dv_budget

    :param dv_budget: Delta V budget of the mission
    :type dv_budget: float
    :param rdvz: Whether to consider rendezvous or flyby
    :type rdvz: bool
    :param df: dataframe with the ISO data to consider, defaults to None
    :type df: pd.DataFrame | None, optional
    :return: fraction of ISOs in dataframe that are reachable with the given dv budget
    :rtype: float
    '''
    if df is None: df = get_data()
    if not rdvz:
        dv = df['icpt_idv']
    else:
        dv = df['rdvz_idv'] + df['rdvz_rdv']
    dv = dv[dv <= dv_budget]
    return len(dv)/len(df)


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

def study_ISO(ISO:jkat.Orbit, park:jkat.Orbit, detect_t:float, gen_type:str)->dict:
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
    # initial data
    
    # check detection distance/time
    out = {}

    try:
        res = helio_optim(park, ISO, ISO.tp + MAX_MISSION_TIME*YEAR,MAX_BOOST_DV)
        out = ({
            'h_tdv': res['dv0'],
            'h_idv': res['dv1'],
            'h_rdv': res['dv2'],
            'h_ts' : (res['ts']-detect_t)/DAY,
            'h_te' : (res['te']-detect_t)/DAY,
            'h_r' : res['r']/AU,
            'h_rad_angle' : m.degrees(res['radial']),
            'h_rad_dv': res['rad_burn'],
            'h_max_boost' : MAX_BOOST_DV
        })
    except(ArithmeticError, ValueError): pass # no intercept :(


    return out



def study_batch(gen_type:str='', longp_num:int = 45)->pd.DataFrame:
    '''generate a batch of ISOs, then study each for several ranges of detect_r
    and then return the resulting dataframe

    :return: dataframe with the results of the study
    :rtype: pd.DataFrame
    '''

    np.seterr(all='ignore') # since we don't care about the errors
    ISOs = get_ISO(gen_type=gen_type)
    # shuffle timings so that does not influence study:
    res_list= []
    for (ISO, detect_t,g_type) in tqdm(ISOs, desc=f"Studying ISOs"):
        detect_r = ISO.r(ISO.f(detect_t))/AU
        out = {"detection_r":detect_r, "periapsis":ISO.periapsis/AU, "magnitude_generation_method": gen_type,
            'time_until_periapsis':(ISO.tp - detect_t)/DAY,
                "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.raan, "arg_p":ISO.argp, "t_p":ISO.tp, 
                "ISO_excess_velocity":ISO.vinf}
        for longp in np.linspace(0,2*np.pi, longp_num):
            out1 = study_ISO(ISO,get_parking(longp),detect_t, g_type)
            if out1 == {}: continue
            name = f"{longp:3.1f}"
            out.update({
                f'h_tdv_{name}' : out1['h_tdv'],
                f'h_idv_{name}' : out1['h_idv'],
                f'h_rdv_{name}' : out1['h_rdv'], 
            })
        out.update(study_ISO(ISO,parking_orbit,detect_t,g_type))
        res_list.append(out)
    return pd.DataFrame(res_list)

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
    # load if applicable
    try:
        data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / PICKLE_NAME)
    except:
        data = pd.DataFrame()
    
    # generate new if applicable:
    if extra_batches > 0:
        new = [data]
        for i in range(extra_batches):
            print('============================================')
            print(f"Generating batch {i+1} of {extra_batches}:")
            print('============================================')
            new.append(study_batch(gen_type, LONGP_NUM))
        data = pd.concat(new,ignore_index=True)
        # save result:
        data.to_pickle(PATH_TO_DATA / PICKLE_NAME)
    # return result:
    return data

def _fix_data():
    '''Debug function to fix issues with the data'''
    data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / PICKLE_NAME)

    # ==== change here ====
    # # get the heliocentric values
    # np.seterr(all="ignore")
    # try:
    #     for i,row in tqdm(data.iterrows(), desc="study helio",total=len(data)):
    #         # if 'h_max_boost' in row: continue
    #         ISO, detect_t,g_type = recreate_ISO(row)
    #         out = study_helio(ISO,detect_t,g_type)
    #         for key, val in out.items():
    #             if key.startswith('h'):
    #                 data.loc[i, key] = val
    # finally: data.to_pickle(PATH_TO_DATA / PICKLE_NAME)

    # =====================

    
def dv_histogram(rdvz:bool,printing:bool = False,df:pd.DataFrame|None=None, **kwargs):
    '''generate a probability density histogram of the delta v requirements

    :param rdvz: Whether to consider rendezvous or flyby
    :type rdvz: bool
    :param printing: whether to print out CDF values for several dv values, defaults to False
    :type printing: bool, optional
    :param df: dataframe with the ISO data to consider, defaults to None
    :type df: pd.DataFrame | None, optional
    '''
    # get right cols
    if df is None: df = get_data()
    if not rdvz:
        dv = df['icpt_idv']
    else:
        dv = df['rdvz_idv'] + df['rdvz_rdv']
    plt.hist(dv,bins=50, range=(0,100), density=True, edgecolor='k', alpha=0.65, histtype="stepfilled", **kwargs)
    plt.xlabel(r"$\Delta V$ requirement")
    plt.ylabel("Probability Density")
    # plt.title(f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if rdvz else "intercept"}\n(Normalization includes unreachable ISOs)")
    if printing:
        func = lambda x: (dv_below_budget(x,rdvz,df))*100
        print("Portion below:")
        print(f"5 km/s: {func(5):.2f}%")
        print(f"10 km/s: {func(10):.2f}%")
        print(f"15 km/s: {func(15):.2f}%")
        print(f"20 km/s: {func(20):.2f}%")
        print(f"40 km/s: {func(40):.2f}%")

def distance_histogram(df:pd.DataFrame, **kwargs):
    '''USE as reference for histograms

    :param df: _description_
    :type df: pd.DataFrame
    '''
    plt.hist(df['detection_r'], bins=20, density=True, **kwargs)
    plt.title("Heliocentric altitude at time of detection probability distribution")
    plt.xlabel("Heliocentric altitude (AU)")
    plt.ylabel("probability density")

def probability_map(df:pd.DataFrame, rdvz:bool, guesses:bool = True, num:int=0):
    '''Generate a probability make of dv_budget against number of detected ISOs

    :param df: dataframe with the ISO data to consider, defaults to None
    :type df: pd.DataFrame
    :param rdvz: Whether to consider rendezvous or flyby
    :type rdvz: bool
    :param guesses: whether to plot guesses on N from the literature, defaults to True
    :type guesses: bool, optional
    '''

    Ezell_Loeb_avg_per_annum = 5
    Hoover_seligman_payne_per_annum = 14
    Marceta_seligman_per_annum = 35
    years = 10 

    EL_N = Ezell_Loeb_avg_per_annum * years
    HSP_N = Hoover_seligman_payne_per_annum * years
    MS_N = Marceta_seligman_per_annum * years
    

    N_range = np.arange(10,MS_N + 30,5)
    V_range =np.arange(1,25, 0.2)
    NN, VV = np.meshgrid(N_range,V_range)
    F = lambda v,n: mission_success_probability(v,n,rdvz,df)
    PP = np.vectorize(F)(VV,NN)
    plt.imshow(PP,origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    if num != 1:
        plt.colorbar(location="right", label=r"$P_s$")
    CS = plt.contour(PP,levels=[0.5,0.9,0.99],origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]), colors='k')
    plt.clabel(CS, fmt=lambda x: f"{x*100:.0f}%")
    if num > 1:
        plt.xlabel(r'$N$')
    plt.ylabel(r'$\Delta V$ budget [km/s]')
    
    # plt.title(f"Probability map for {"rendezvous" if rdvz else "intercept"}\nAnd estimated ISO detections during {years} year mission")
    if guesses:
        plt.axvline(EL_N,ls='--', color="gray")
        plt.text(EL_N+1, np.average(V_range)+3, "Ezell, Loeb mean", color="gray")
        plt.axvline(HSP_N,ls='--', color="gray")
        plt.text(HSP_N+1, np.average(V_range), "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.axvline(MS_N,ls='--', color="gray")
        plt.text(MS_N-1, np.average(V_range)-3, "Marčeta, Seligman mean", ha="right", color="gray")

    plt.gca().set_aspect(N_range[-1]/V_range[-1])
    return PP, N_range, V_range

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
    res = helio_optim(parking_orbit,ISO, ISO.tp + MAX_MISSION_TIME*YEAR, MAX_BOOST_DV)
    ROT:jkat.Orbit = res['ob']
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


def plots_for_iso_detection():
    '''function to generate the plots and numbers for the iso detection chapter in the LaTeX'''

    # want:
    # detection distance, detection time, distribution of them, for both omuamua-like and borisov like
    # ratio detected
    df = get_data()
    dfb = df[df['magnitude_generation_method']=='atlas-borisov']
    dfo = df[df['magnitude_generation_method']=='omuamua']
    print('DATA:\n')
    print(f'fraction omuamua: {len(dfo)/len(df)*100:.3f}%, number omuamua: {len(dfo)}')
    print(f'fraction borisov: {len(dfb)/len(df)*100:.3f}%, number borisov: {len(dfb)}')
    print()



    # distance:
    plt.subplot(1,2,1)
    plt.title('Cometary')
    br = dfb['detection_r']
    plt.hist(br, bins=20, density=True, color='b', edgecolor='k', alpha=0.65, histtype="stepfilled")
    plt.axvline(np.average(br),color='k', linestyle='dashed', linewidth=1)
    plt.xlabel("Heliocentric distance at time of detection [AU]")
    plt.ylabel("Probability density")
    
    print(f'borisov average: {np.average(br)}')
    # plt.show()
    plt.subplot(1,2,2)
    plt.title('Asteroidal')

    # distance:
    br = dfo['detection_r']
    plt.hist(br, bins=20, density=True, color='y', edgecolor='k', alpha=0.65, histtype="stepfilled")
    plt.axvline(np.average(br),color='k', linestyle='dashed', linewidth=1)
    plt.xlabel("Heliocentric distance at time of detection [AU]")
    plt.ylabel("Probability density")
    
    print(f'omuamua average: {np.average(br)}')
    plt.show()

    # time:
    plt.subplot(1,2,1)
    plt.title('Cometary')
    br = dfb['time_until_periapsis']
    plt.hist(br, bins=20, density=True, color='b', edgecolor='k', alpha=0.65, histtype="stepfilled")
    plt.axvline(np.average(br),color='k', linestyle='dashed', linewidth=1)
    plt.xlabel("Time until perihelion after detection [days]")
    plt.ylabel("Probability density")
    
    print(f'borisov average: {np.average(br)}')
    plt.subplot(1,2,2)
    plt.title('Asteroidal')

    # time:
    br = dfo['time_until_periapsis']
    plt.hist(br, bins=20, density=True, color='y', edgecolor='k', alpha=0.65, histtype="stepfilled")
    plt.axvline(np.average(br),color='k', linestyle='dashed', linewidth=1)
    plt.xlabel("Time until perihelion after detection [days]")
    plt.ylabel("Probability density")
    
    print(f'omuamua average: {np.average(br)}')
    plt.show()

def plots_for_dv_histogram():


    df = get_data()
    dvi = df['h_idv']
    dvr = df['h_ion_total']
    plt.hist(dvi, bins=40, range=(0,100), density=True, color=('r'), alpha=0.65, histtype="stepfilled", edgecolor='k', label="boost dv")
    plt.hist(dvr, bins=40, range=(0,100), density=True, color=('orange'), alpha=0.65, histtype="stepfilled", edgecolor='k', label="ion dv")
    plt.xlabel(r"$\Delta V$ requirement")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()
   
def plots_for_probability_map():
    df = get_data()
    plt.subplot(2,1,1)
    plt.title("Flyby")
    PPi, N_range, V_range = probability_map(df,False,False,1)
    plt.subplot(2,1,2)
    plt.title("Rendezvous")
    PPr, _, _ = probability_map(df,True,False,2)

    # Chosen N:
    N = 150
    idx_n = 0
    while N_range[idx_n] < N: idx_n +=1

    #flyby:
    P_range = PPi[:,idx_n]
    idx_v = 0
    while P_range[idx_v] < 0.9: idx_v += 1
    Vi = V_range[idx_v]

    #rendezvous:
    P_range = PPr[:,idx_n]
    idx_v = 0
    while P_range[idx_v] < 0.9: idx_v += 1
    Vr = V_range[idx_v]
    print(f"for 90%, intercept needs: {Vi:.4f} km/s dV and rendezvous needs: {Vr:.4f} km/s dV")
    


    plt.show()

def longp_graph(df:pd.DataFrame, fraction:float, longp_num:int = 45):

    

    pp = []
    vv = []

    for longp in np.linspace(0,2*np.pi, longp_num):
        pp.append(longp)
        name = f"{longp:3.1f}"
        v = df[f'h_idv_{name}']
        v = v.sort_values(ignore_index=True)
        n = m.ceil(len(v)*fraction)
        vreq = v[n]
        vv.append(vreq)
        print(f" for {name}: number required is: {n} out of {len(v)}, corresponding to: {vreq:3.2f} km/s")

    plt.polar(pp,vv)
    plt.title('oberth dv required per longitude of periapsis')
    plt.show()
        



def run_in_background():
    '''run forever generating new datapoints'''
    while True: 
        df = get_data(1)
        print("Current # of rows:")
        print(len(df))
        print('---------\n')


# ======= plotting and analysis ========

if __name__ == "__main__":

    df = get_data(0)
    # prob_needed = 0.0152 # N = 150
    prob_needed = 0.0076 # N = 300

    longp_graph(df,prob_needed, LONGP_NUM)

    # plots_for_probability_map()
    print(df)
    # df = df[pd.notna(df['h_tdv'])]
    df = df.sort_values('h_idv', ignore_index=True)
    print(df[["h_tdv", "h_idv","h_rdv", "h_r", "h_ts", "h_te", 'h_rad_angle','h_rad_dv',"periapsis"]])

    

    n_needed = int(np.floor(len(df) * prob_needed)) # 0.76% from N =300
    assert (n_needed + 1) / prob_needed > len(df)
    print(f'dv needed (rough): {df.iloc[n_needed]['h_idv']} --- {df.iloc[n_needed+1]['h_idv']} km/s')
    
    plt.hist(df['h_idv'],bins=len(df)//50)
    plt.show()
    # plot_from_row(df.iloc[1], 10*AU)

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
