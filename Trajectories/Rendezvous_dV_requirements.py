'''
Script for generating a distributions of dV expected for the ISO intercept
and from that the mission success chance
missing is a proper distribution of N, which requires more research in the literature
'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.orbit import Orbit, trajectory_optimizer, orbit_from_lambert_transfer, plot_orbit, get_solar_system_ax
from src.get_ISO import get_ISO
from src.examples import Earth, Jupiter
from src.utilities import AU, YEAR, SGP_SUN, DAY
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
import pandas as pd

# SETTINGS:

PATH_TO_DATA = Path(__file__).parent.parent / "data" 
PICKLE_NAME = "dvreq.pic"
icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
MAX_MISSION_TIME = 20 # [years]

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
col_names = ["detection_r", "periapsis", "magitude_generation_method",
             "parameter", "e", "i", "RAAN", "arg_p", "t_p", 
             "icpt_idv", "icpt_rdv", "icpt_r", "icpt_t_launch", "icpt_t_arrival",
             "rdvz_idv", "rdvz_rdv", "rdvz_r", "rdvz_t_launch", "rdvz_t_arrival",
             ]

def study_ISO(ISO:Orbit, detect_t:float, gen_type:str)->dict:
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
    detect_r = ISO.polar_equation(ISO.time_to_theta(detect_t))/AU
    out = {"detection_r":detect_r, "periapsis":ISO.periapsis/AU, "magitude_generation_method": gen_type,
             "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.RAAN, "arg_p":ISO.arg_p, "t_p":ISO.t_p, }
    
    # check detection distance/time
    
    # intercept:
    try:
        insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(Earth,ISO,detect_t,detect_t+MAX_MISSION_TIME*YEAR, **icpt_weights)
        out.update({
            "icpt_idv":insert_dv, 
            "icpt_rdv": rdvz_dv, 
            "icpt_r": er/AU, 
            "icpt_t_launch":(st - detect_t)/DAY, 
            "icpt_t_arrival":(et - detect_t)/DAY
        })
    except (ArithmeticError,ValueError):
        pass # no intercept :(
    # rendezvous:
    try:
        insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(Earth,ISO,detect_t,detect_t+MAX_MISSION_TIME*YEAR, **rdvz_weights)
        out.update({
            "rdvz_idv":insert_dv, 
            "rdvz_rdv": rdvz_dv, 
            "rdvz_total": insert_dv + rdvz_dv,
            "rdvz_r": er/AU, 
            "rdvz_t_launch":(st - detect_t)/DAY, 
            "rdvz_t_arrival":(et - detect_t)/DAY
        })
    except (ArithmeticError,ValueError):
        pass # no rendezvous :(

    return out

def study_batch(gen_type:str='')->pd.DataFrame:
    '''generate a batch of ISOs, then study each for several ranges of detect_r
    and then return the resulting dataframe

    :return: dataframe with the results of the study
    :rtype: pd.DataFrame
    '''

    np.seterr(divide='ignore', invalid='ignore') # since we don't care about the errors
    ISOs = get_ISO(gen_type=gen_type)
    # shuffle timings so that does not influence study:
    res_list= []
    for (ISO, detect_t,g_type) in tqdm(ISOs, desc=f"Studying ISOs"):
        res_list.append(study_ISO(ISO,detect_t,g_type))
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
            new.append(study_batch(gen_type))
        data = pd.concat(new,ignore_index=True)
        # save result:
        data.to_pickle(PATH_TO_DATA / PICKLE_NAME)
    # return result:
    return data

def _fix_data():
    '''Debug function to fix issues with the data'''
    data:pd.DataFrame = pd.read_pickle(PATH_TO_DATA / PICKLE_NAME)

    # ==== change here ====

    # data["icpt_t_arrival"] = data["icpt_t_arrival"]/DAY
    # data["icpt_t_launch"] = data["icpt_t_launch"]/DAY
    # data["rdvz_t_launch"] = data["rdvz_t_launch"]/DAY
    # data["rdvz_t_arrival"] = data["rdvz_t_arrival"]/DAY

    # =====================

    data.to_pickle(PATH_TO_DATA / PICKLE_NAME)

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
    plt.hist(dv,bins=100, range=(0,100), density=True, **kwargs)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.title(f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if rdvz else "intercept"}\n(Normalization includes unreachable ISOs)")
    if printing:
        func = lambda x: (dv_below_budget(x,rdvz,df))*100
        print("Portion below:")
        print(f"5 km/s: {func(5):.2f}%")
        print(f"10 km/s: {func(10):.2f}%")
        print(f"15 km/s: {func(15):.2f}%")
        print(f"20 km/s: {func(20):.2f}%")
        print(f"40 km/s: {func(40):.2f}%")

def probability_map(df:pd.DataFrame, rdvz:bool, guesses:bool = True, show:bool=True):
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
    V_range =np.arange(4,50)
    NN, VV = np.meshgrid(N_range,V_range)
    F = lambda v,n: mission_success_probability(v,n,rdvz,df)
    PP = np.vectorize(F)(VV,NN)
    plt.imshow(PP,origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    plt.colorbar(location="right", label="Probability of mission success")
    CS = plt.contour(PP,levels=[0.9],origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    plt.clabel(CS, fmt=lambda x: f"{x*100:.0f}%")
    plt.ylabel('Delta V budget (km/s)')
    plt.xlabel('number of ISOs during mission time')
    plt.title(f"Probability map for {"rendezvous" if rdvz else "intercept"}\nAnd estimated ISO detections during {years} year mission")
    if guesses:
        plt.plot([EL_N,EL_N],[5,48], ls='--', color="gray")
        plt.text(EL_N+1, 40, "Ezell, Loeb mean", color="gray")
        plt.plot([HSP_N,HSP_N],[5,48], ls='--', color="gray")
        plt.text(HSP_N+1, 30, "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.plot([MS_N,MS_N],[5,48], ls='--', color="gray")
        plt.text(MS_N-1, 20, "Marčeta, Seligman mean", ha="right", color="gray")


def plot_from_row(ax, row:pd.Series, max_r:float=m.inf):
    '''Plot a 3d representation of the values of a row, plots both rendezvous and intercept trajectories

    :param ax: matplotlib axes to plot in, needs to be 3d
    :type ax: _type_
    :param row: row to plot
    :type row: pd.Series
    :param max_r: max distance to plot, in AU, if omitted plots up to furthest intercept
    :type max_r: float, optional
    '''

    # extract orbit:
    ISO = Orbit(
        row['parameter'],
        row['e'],
        row['i'],
        row['RAAN'],
        row['arg_p'],
        row['t_p'],
        SGP_SUN
    )
    detect_r = row['detection_r']
    max_r = min(max_r,max(row["icpt_r"], row["rdvz_r"]))
    
    
    t_detect = ISO.theta_to_time(-ISO.crosses_altitude(detect_r*AU)) # type:ignore
    icpt_s = t_detect + row["icpt_t_launch"]
    icpt_e = t_detect + row["icpt_t_arrival"]
    rdvz_s = t_detect + row["rdvz_t_launch"]
    rdvz_e = t_detect + row["rdvz_t_arrival"]
    max_t = max(rdvz_e, icpt_e)

    # get the intercept:
    ICPT = orbit_from_lambert_transfer(Earth,ISO,icpt_s,icpt_e)
    plot_orbit(ax, ICPT, (icpt_s,icpt_e), max_alt=AU*max_r + 100, label="Flyby trajectory") # slight margin on max alt

    # get the rendezvouz:
    RDVZ = orbit_from_lambert_transfer(Earth,ISO,rdvz_s,rdvz_e)
    plot_orbit(ax, RDVZ, (rdvz_s,rdvz_e), max_alt=AU*max_r + 100, label="Rendezvous trajectory")

    # plot earth, jupiter and iso:
    plot_orbit(ax,Earth, max_t, label="Earth", color="Blue")
    plot_orbit(ax,ISO,max_t, max_alt=AU*max_r + 100, label="ISO")
    plot_orbit(ax,Jupiter, max_t, label="Jupiter", color="orange")

    # printing:
    print(f'intercept:\nlaunches: {row["icpt_t_launch"]/DAY:.2f} days after detection, arrives {row["icpt_t_arrival"]/DAY:.2f} days after detection at a distance of {row["icpt_r"]} AU')
    print(f"initial delta v cost is: {row["icpt_idv"]:.2f} km/s, and relative velocity at intercept is {row['icpt_rdv']} km/s\n")

    print(f'Rendezvous:\nlaunches: {row["rdvz_t_launch"]/DAY:.2f} days after detection, arrives {row["rdvz_t_arrival"]/DAY:.2f} days after detection at a distance of {row["rdvz_r"]} AU')
    print(f"initial delta v cost is: {row["rdvz_idv"]:.2f} km/s, and relative velocity at intercept is {row['rdvz_rdv']} km/s, for a total delta v of {row["rdvz_total"]:.2f} km/s")


# ======= plotting and analysis ========

if __name__ == "__main__":

    df = get_data(1, "omuamua")
    print(df)
    input()
    dv_histogram(False, df=df)
    plt.show()
    dv_histogram(True, df=df)
    plt.show()

    # df_om = df[df['magitude_generation_method'] == 'omuamua']
    # df_bori = df[df['magitude_generation_method'] == 'atlas-borisov']
    # # probability_map(df_om, False)
    # probability_map(df_bori, False)
    
    # print(len(df_om))
    # print(len(df_bori))
    # plt.hist(df[df['magitude_generation_method'] == 'omuamua']['detection_r'],density=True, label="omuamua-like",fill=False, edgecolor="blue")
    # plt.hist(df[df['magitude_generation_method'] == 'atlas-borisov']['detection_r'],density=True, label="atlas-borisov-like",fill=False, edgecolor="red")
    # plt.title("detection range probability distribution(AU)")
    # plt.legend()
    # plt.show()


    while True:
        
        df = get_data(1,gen_type="omuamua")
        print("Current # of rows:")
        print(len(df[df['magitude_generation_method'] == 'omuamua']))
        print('---------\n')






    # detect_r = 5
    # dv_histogram(detect_r,False)
    # print(f"fraction below 5 km/s: {dv_below_budget(5,detect_r,False):.3f}\n10 km/s: {dv_below_budget(10,detect_r,False):.3f}\n20 km/s: {dv_below_budget(20,detect_r,False):.3f}\n40km/s: {dv_below_budget(40,detect_r,False):.3f}")
    # plt.show()
    
    # data = get_data()
    # data = data.sort_values("icpt_idv", ignore_index=True)
    # # print(data[["rdvz_idv", "rdvz_rdv", "rdvz_r", "rdvz_t_launch", "rdvz_t_arrival","generated_rm", "detection_r", "periapsis"]])

    # row = data.iloc[0]
    # # print(row)
    # ax = get_solar_system_ax()
    # plot_from_row(ax,row) # type:ignore
    # plt.legend()
    # plt.show()
