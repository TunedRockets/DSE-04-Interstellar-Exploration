'''
Script for generating a distributions of dV expected for the ISO intercept


'''
from src2.orbit import Orbit, oberth_effect_optimzer, plot_orbit, orbit_from_lambert, orbit_from_rv, orbit_from_ephemeris
from src2.get_ISO import get_ISO, load_ISOs
from src2.examples import Mercury, Venus, Mars, Earth, Jupiter, Saturn, Uranus, Neptune, Pluto, get_solar_system_ax
from src2.utilities import AU, YEAR, SGP_SUN, DAY
import pandas as pd
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent.resolve()))

PLOT= True

if PLOT:
    import matplotlib as mpl
    mpl.use('TkAgg')


import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
from pathlib import Path
import random
# plt.xkcd(scale=1, length=100, randomness=2)


rdvz = True

aphelion = 1*5.4507 * AU # Jupiter aphelion
solar_radius = 696_340
perihelion = 10*solar_radius
semi_major_axis = (aphelion + perihelion) / 2
eccentricity = (aphelion - perihelion) / (aphelion + perihelion)

origin_124 = orbit_from_ephemeris(
            semi_major_axis,
            eccentricity,
            m.radians(1.303),
            m.radians(100.46457166),
            m.radians(124.14), # Found longitude of periapsis
            m.radians(100.464),
            SGP_SUN
        )

PATH_TO_DATA = Path(__file__).parent.parent / "data_oberth"
PICKLE_NAME = "ISOdata_oberth.pic"

def add_dv_hist(origin, max_time, weights, N, PLOT=False, lon_per=None)->None:
    '''Adds to the dv histogram for the different weights'''
    np.seterr(all="ignore")
    # np.seterr(all="raise")
    if lon_per is None:
        lon_per_str=""
    else:
        lon_per_str=str(round(np.degrees(lon_per)))

    path = Path(__file__).parent.parent / "data_oberth" / (f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},"+lon_per_str)
    
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            count = int(lines[0])
            hist = [int(x) for x in lines[1:]]
            if count>=N:
                return

    except:
        hist = [0 for _ in range(100)]
        count = 0


    # ISOs = get_ISO() # sample of ISOs
    ISOs = load_ISOs("10000_ISOs_new.pkl", plot=False)

    # --- trim if too many ---
    if len(ISOs) > N:
        ISOs = random.sample(ISOs, N)

    # --- otherwise top up ---
    while len(ISOs) < N:
        ISOs.extend(get_ISO())

    # optional safety trim (in case last batch overshoots)
    if len(ISOs) > N:
        ISOs = random.sample(ISOs, N)

    count += len(ISOs)

    for ISO in tqdm(ISOs,desc="studying ISOs"):

        detect_time = ISO[1]
        if detect_time is None:
            continue
        detect_theta = ISO[0].time_to_theta(detect_time)
        try:
            # ===============================
            # Oberth optimization at periapsis
            # ===============================
            theta_pe = 0
            # tp = origin.theta_to_time(theta_pe)
            or_period = origin.period
            tp = random.uniform(0.5*or_period, 1.5*or_period)

            rp_vec, vp_vec = origin.theta_to_rv(theta_pe)
            vp_mag = np.linalg.norm(vp_vec)

            min_time = 100

            insert_dv, rdvz_dv, transfer_orbit, st, et, er = oberth_effect_optimzer(
                ISO[0],
                rp_vec,
                vp_mag,
                tp,
                min_time,
                max_time,
                optimize_rendezvous=(weights["w_relv"] > 0),
                period=or_period,
                detect_time=detect_time,
                periods=None
                # tp_window_width=10*YEAR/365
            )

            # ===============================
            # Compute required periapsis direction
            # ===============================
            r_req, v_req = transfer_orbit.theta_to_rv(theta_pe)
            v_req_hat = v_req / np.linalg.norm(v_req)

            # current orbit periapsis direction
            r0, v0 = origin.theta_to_rv(theta_pe)
            v0_hat = v0 / np.linalg.norm(v0)
            #
            # # ===============================
            # # Angle between directions
            # # ===============================
            cos_dtheta = np.clip(np.dot(v0_hat, v_req_hat), -1, 1)
            delta = m.acos(cos_dtheta)
            total_dv=insert_dv
            if weights['w_relv']>0:
                total_dv +=rdvz_dv
            #
            # # ===============================
            # # Apoapsis velocity
            # # ===============================
            # r_a = origin.apoapsis
            # a = origin.a
            # mu = origin.sgp
            #
            # v_ap = m.sqrt(mu * (2 / r_a - 1 / a))
            #
            # # ===============================
            # # Plane change Δv at apoapsis
            # # ===============================
            # dv_plane = 2 * v_ap * m.sin(delta / 2)
            #
            # # ===============================
            # # Total insertion Δv
            # # ===============================
            # insert_dv += dv_plane
            # ===============================
            # Reconstruct rotated orbit (after apoapsis burn)
            # ===============================

            # rotation axis from current to required direction
            axis = np.cross(v0_hat, v_req_hat)
            norm = np.linalg.norm(axis)

            if norm < 1e-10:
                axis = np.array([0, 0, 1])  # fallback
            else:
                axis /= norm

            def rotate(vec):
                return (
                        vec * m.cos(delta) +
                        np.cross(axis, vec) * m.sin(delta) +
                        axis * np.dot(axis, vec) * (1 - m.cos(delta))
                )

            # get apoapsis state
            theta_ap = m.pi
            t_ap = origin.theta_to_time(theta_ap)
            r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

            # rotate state
            r_rot = rotate(r_ap)
            v_rot = rotate(v_ap_vec)
            inc_dv = np.linalg.norm((v_rot - v_ap_vec))
            total_dv += inc_dv

        except:
            continue

        total_dv = round(total_dv)
        if PLOT and total_dv < 20:
            print("Transfer a: ", transfer_orbit.a)
            print("Transfer e: ", transfer_orbit.e)
            print()

            # ==== plotting ====
            # fig = plt.figure()
            ax = get_solar_system_ax()

            # plot original orbit
            plot_orbit(ax, origin, time=detect_time, ThreeDee=True, label="Original")
            # rebuild orbit
            try:
                origin_rot = orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
                # origin_rot.link_time_and_theta(theta_ap, t_ap)
                # origin_rot.normalize()
                # plot rotated orbit
                plot_orbit(ax, origin_rot, time=detect_time, ThreeDee=True, label="Rotated")
            except Exception as e:
                print()
                print("Rotated orbit rebuilding failed: ", e)
                print("Delta V: ", total_dv)


            try:
                plot_orbit(ax, transfer_orbit, time=et, ThreeDee=True, label="Transfer", max_alt=(100*AU))
            except:
                pass  # lambert sometimes fails

            # plot ISO, earth and jupiter orbit for context
            plot_orbit(ax, ISO[0], time=et, ThreeDee=True, label="ISO", max_alt=(100*AU))
            plot_orbit(ax, Earth, time=detect_time, ThreeDee=True, label="Earth")
            plot_orbit(ax, Jupiter, time=detect_time, ThreeDee=True, label="Jupiter")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.axis("equal")
            textstr = (
                f"ΔV inclination: {inc_dv:.2f} km/s\n"
                f"ΔV insert: {insert_dv:.2f} km/s\n"
                f"ΔV rendezvous: {rdvz_dv:.2f} km/s\n"
                f"ΔV total: {total_dv:.2f} km/s\n"
                f"Intercept distance: {np.linalg.norm(ISO[0].time_to_rv(et)[0])/AU:.2f} AU\n"
                f"Intercept time: {(et-detect_time)/YEAR:.2f} years\n"
            )

            ax.text2D(0.02, 0.98, textstr,
                      transform=ax.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.legend()

            plt.show()
        if total_dv > 99 or total_dv < 0 or not np.isfinite(total_dv): continue
        else:
            hist[round(total_dv)] += 1
    
    # Save
    path = Path(__file__).parent.parent / "data_oberth" / (f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},"+lon_per_str)
    with open(path, "w") as file:
        file.write(str(count) + '\n')
        file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(weights, lon_per=None)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    if lon_per is None:
        lon_per_str = ""
    else:
        lon_per_str = str(round(np.degrees(lon_per)))
    path = Path(__file__).parent.parent / "data_oberth" / (
                f"dVhist-{weights["w_insertion"]},{weights["w_relv"]}," + lon_per_str)

    with open(path, "r") as file:
        lines = file.readlines()
        count = int(lines[0])
        hist = [int(x) for x in lines[1:]]
    return [x/count for x in hist]


def mission_success_probability(dV_budget:int, N:int, weight:dict, lon_per=None)->float:
    '''
    Generates total mission probability for the given scenario.
    :param detection_distance: distance (in AU) from the sun that ISOs are detected
    :type: float
    :param dV_buget: total mission dV budget
    :type: int
    :param N: Number of ISOs during the mission
    :type: float
    :param weight: optimizer weight, i.e. an intercept or rendezvouz
    :type: dict
    '''
    p_ISO = ISO_probability(dV_budget, weight, lon_per=lon_per)
    p_least_one = 1-(1-p_ISO)**N
    return p_least_one

def ISO_probability(dV_budget:int, weight,lon_per)->float:
    '''calculate individual chance of success for given dv budget and detection distance,
    currently only works with integer dV budgets

    :param detection_distance: distance (in AU) from the sun that ISOs are detected
    :type: float
    :param dV_buget: total mission dV budget
    :type: int
    :param weight: optimizer weight, i.e. an intercept or rendezvouz
    :type: dict'''
    hist = get_dv_hist(weight,lon_per=lon_per)
    return np.sum(hist[:m.floor(dV_budget)+1])

def probability_map(weight: dict, guesses: bool = True, show: bool = True, lon_per=None):
    '''generate probability map of N over dV,'''

    Ezell_Loeb_avg_per_annum = 5
    Hoover_seligman_payne_per_annum = 14
    Marceta_seligman_per_annum = 35
    years = 10

    EL_N = Ezell_Loeb_avg_per_annum * years
    HSP_N = Hoover_seligman_payne_per_annum * years
    MS_N = Marceta_seligman_per_annum * years

    N_range = np.arange(10, MS_N + 30, 5)
    V_range = np.arange(4, 50)
    NN, VV = np.meshgrid(N_range, V_range)
    PP = np.vectorize(mission_success_probability)(VV, NN, weight, lon_per=lon_per)
    plt.imshow(PP, origin="lower", aspect="auto", extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.colorbar(location="right", label="Probability of mission success")
    CS = plt.contour(PP, levels=[0.9], origin="lower", aspect="auto",
                     extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.clabel(CS, fmt=lambda x: f"{x * 100:.0f}%")
    plt.ylabel('Delta V budget (km/s)')
    plt.xlabel('number of ISOs during mission time')
    plt.title(
        f"Probability map for {"rendezvous" if weight["w_relv"] else "intercept"}\nAnd estimated ISO detections during {years} year mission, lon per {np.degrees(lon_per)} deg")
    if guesses:
        plt.plot([EL_N, EL_N], [5, 48], ls='--', color="gray")
        plt.text(EL_N + 1, 40, "Ezell, Loeb mean", color="gray")
        plt.plot([HSP_N, HSP_N], [5, 48], ls='--', color="gray")
        plt.text(HSP_N + 1, 30, "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.plot([MS_N, MS_N], [5, 48], ls='--', color="gray")
        plt.text(MS_N - 1, 20, "Marčeta, Seligman mean", ha="right", color="gray")
    if show:
        plt.show()

def distribution_histogram(weight:dict,  show:bool=True, lon_per=None):
    '''generate the histogram of the dV requirements'''

    hist = get_dv_hist(weight, lon_per=lon_per)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.title(f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if weight["w_relv"] else "intercept"}\n. (Normalization includes unreachable ISOs)")
    if show:
        plt.show()

def dv_for_confidence(weight, N, target_prob, lon_per=None, dv_max=100):
    for dv in range(dv_max + 1):
        p = mission_success_probability(dv, N, weight, lon_per=lon_per)
        if p >= target_prob:
            return dv
    return np.nan  # if not achievable

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
             ]


def study_ISO(ISO: Orbit, detect_t: float, gen_type: str, origin: Orbit=origin_124) -> dict:
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
        :param origin: the origin parking orbit
    :type origin: Orbit
    '''
    # initial data
    detect_r = ISO.polar_equation(ISO.time_to_theta(detect_t)) / AU
    out = {"detection_r": detect_r, "periapsis": ISO.periapsis / AU, "magnitude_generation_method": gen_type,
           'time_until_periapsis': (ISO.t_p - detect_t) / DAY,
           "parameter": ISO.p, "e": ISO.e, "i": ISO.i, "RAAN": ISO.RAAN, "arg_p": ISO.arg_p, "t_p": ISO.t_p, }

    # check detection distance/time

    t_p = random.uniform(0.5 * origin.period, 1.5 * origin.period)

    # intercept:
    try:
        # insert_dv, rdvz_dv, st, et, er = trajectory_optimizer(Earth, ISO, detect_t, detect_t + MAX_MISSION_TIME * YEAR,
                                                              # **icpt_weights)
        insert_dv, rdvz_dv, transfer_o, st, et, er = oberth_effect_optimzer(ISO, origin.theta_to_rv(0)[0], np.linalg.norm(origin.theta_to_rv(0)[1]), t_p, 0, 40*YEAR, period=origin.period, optimize_rendezvous=False, detect_time=detect_t)
        out.update({
            "icpt_idv": insert_dv,
            "icpt_rdv": rdvz_dv,
            "icpt_r": er / AU,
            "icpt_t_launch": t_p / DAY,
            "icpt_t_arrival": (et - detect_t) / DAY
        })
    except (ArithmeticError, ValueError, RuntimeError) as e:
        # print("No intercept lmao: ", e)
        # raise e
        pass  # no intercept :(
    # rendezvous:
    try:
        # insert_dv, rdvz_dv, st, et, er = trajectory_optimizer(Earth, ISO, detect_t, detect_t + MAX_MISSION_TIME * YEAR,
        #                                                       **rdvz_weights)
        insert_dv, rdvz_dv, transfer_o, st, et, er = oberth_effect_optimzer(ISO, origin.theta_to_rv(0)[0], np.linalg.norm(origin.theta_to_rv(0)[1]), t_p, 0, 40*YEAR, period=origin.period, optimize_rendezvous=True, detect_time=detect_t)
        # ===============================
        # Compute required periapsis direction
        # ===============================
        r_req, v_req = transfer_o.theta_to_rv(0)
        v_req_hat = v_req / np.linalg.norm(v_req)

        # current orbit periapsis direction
        r0, v0 = origin.theta_to_rv(0)
        v0_hat = v0 / np.linalg.norm(v0)
        #
        # # ===============================
        # # Angle between directions
        # # ===============================
        cos_dtheta = np.clip(np.dot(v0_hat, v_req_hat), -1, 1)
        delta = m.acos(cos_dtheta)

        # rotation axis from current to required direction
        axis = np.cross(v0_hat, v_req_hat)
        norm = np.linalg.norm(axis)

        if norm < 1e-10:
            axis = np.array([0, 0, 1])  # fallback
        else:
            axis /= norm

        def rotate(vec):
            return (
                    vec * m.cos(delta) +
                    np.cross(axis, vec) * m.sin(delta) +
                    axis * np.dot(axis, vec) * (1 - m.cos(delta))
            )

        # get apoapsis state
        theta_ap = m.pi
        t_ap = origin.theta_to_time(theta_ap)
        r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

        # rotate state
        r_rot = rotate(r_ap)
        v_rot = rotate(v_ap_vec)
        inc_dv = np.linalg.norm((v_rot - v_ap_vec))
        insert_dv += inc_dv
        out.update({
            "rdvz_idv": insert_dv,
            "rdvz_rdv": rdvz_dv,
            "rdvz_total": insert_dv + rdvz_dv,
            "rdvz_r": er / AU,
            "rdvz_t_launch": (st - detect_t) / DAY,
            "rdvz_t_arrival": (et - detect_t) / DAY
        })
    except (ArithmeticError, ValueError, RuntimeError) as e:
        # print("No intercept lmao: ", e)
        # raise e
        pass  # no intercept :(

    return out


def study_batch(gen_type: str = '') -> pd.DataFrame:
    '''generate a batch of ISOs, then study each for several ranges of detect_r
    and then return the resulting dataframe

    :return: dataframe with the results of the study
    :rtype: pd.DataFrame
    '''

    np.seterr(divide='ignore', invalid='ignore')  # since we don't care about the errors
    ISOs = get_ISO(gen_type=gen_type)
    # shuffle timings so that does not influence study:
    res_list = []
    for (ISO, detect_t, g_type) in tqdm(ISOs, desc=f"Studying ISOs"):
        res_list.append(study_ISO(ISO, detect_t, g_type))
    return pd.DataFrame(res_list)


def get_data(extra_batches: int = 0, gen_type: str = "") -> pd.DataFrame:
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
        data: pd.DataFrame = pd.read_pickle(PATH_TO_DATA / PICKLE_NAME)
    except:
        data = pd.DataFrame()

    # generate new if applicable:
    if extra_batches > 0:
        new = [data]
        for i in range(extra_batches):
            print('============================================')
            print(f"Generating batch {i + 1} of {extra_batches}:")
            print('============================================')
            new.append(study_batch(gen_type))
        data = pd.concat(new, ignore_index=True)
        # save result:
        data.to_pickle(PATH_TO_DATA / PICKLE_NAME)
    # return result:
    return data


def _fix_data():
    '''Debug function to fix issues with the data'''
    data: pd.DataFrame = pd.read_pickle(PATH_TO_DATA / PICKLE_NAME)

    # ==== change here ====
    # data["icpt_t_launch"] = data["icpt_t_launch"]/DAY
    # data["rdvz_t_launch"] = data["rdvz_t_launch"]/DAY
    # data["rdvz_t_arrival"] = data["rdvz_t_arrival"]/DAY

    # =====================

    data.to_pickle(PATH_TO_DATA / PICKLE_NAME)


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

def dv_histogram(rdvz: bool, printing: bool = False, df: pd.DataFrame | None = None, **kwargs):
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
    plt.hist(dv, bins=100, range=(0, 100), density=True, **kwargs)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.title(
        f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if rdvz else "intercept"}\n(Normalization includes unreachable ISOs)")
    if printing:
        func = lambda x: (dv_below_budget(x, rdvz, df)) * 100
        print("Portion below:")
        print(f"5 km/s: {func(5):.2f}%")
        print(f"10 km/s: {func(10):.2f}%")
        print(f"15 km/s: {func(15):.2f}%")
        print(f"20 km/s: {func(20):.2f}%")
        print(f"40 km/s: {func(40):.2f}%")


def mission_success_probability_df(dV_budget:int, N:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
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
    p_ISO = ISO_probability_df(dV_budget, rdvz, df)
    p_least_one = 1-(1-p_ISO)**N
    return p_least_one

def ISO_probability_df(dV_budget:int, rdvz:bool, df:pd.DataFrame|None=None)->float:
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


def probability_map_df(df: pd.DataFrame, rdvz: bool, guesses: bool = True, show: bool = True):
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

    N_range = np.arange(10, MS_N*2, 5)
    V_range = np.arange(4, 50)
    NN, VV = np.meshgrid(N_range, V_range)
    F = lambda v, n: mission_success_probability_df(v, n, rdvz, df)
    PP = np.vectorize(F)(VV, NN)
    plt.imshow(PP, origin="lower", aspect="auto", extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.colorbar(location="right", label="Probability of mission success")
    CS = plt.contour(PP, levels=[0.9], origin="lower", aspect="auto",
                     extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.clabel(CS, fmt=lambda x: f"{x * 100:.0f}%")
    plt.ylabel('Delta V budget (km/s)')
    plt.xlabel('number of ISOs during mission time')
    plt.title(
        f"Probability map for {"rendezvous" if rdvz else "intercept"}\nAnd estimated ISO detections during {years} year mission")
    if guesses:
        plt.plot([EL_N, EL_N], [5, 48], ls='--', color="gray")
        plt.text(EL_N + 1, 40, "Ezell, Loeb mean", color="gray")
        plt.plot([HSP_N, HSP_N], [5, 48], ls='--', color="gray")
        plt.text(HSP_N + 1, 30, "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.plot([MS_N, MS_N], [5, 48], ls='--', color="gray")
        plt.text(MS_N - 1, 20, "Marčeta, Seligman mean", ha="right", color="gray")


def plot_from_row(ax, row: pd.Series, max_r: float = m.inf, origin=origin_124):
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
    max_r = min(max_r, max(row["icpt_r"], row["rdvz_r"]))

    t_detect = ISO.theta_to_time(-ISO.crosses_altitude(detect_r * AU))  # type:ignore
    icpt_s = t_detect + row["icpt_t_launch"] * DAY
    icpt_e = t_detect + row["icpt_t_arrival"] * DAY
    rdvz_s = t_detect + row["rdvz_t_launch"] * DAY
    rdvz_e = t_detect + row["rdvz_t_arrival"] * DAY
    max_t = max(rdvz_e, icpt_e)

    # ===============================
    # Oberth optimization at periapsis
    # ===============================
    theta_pe = 0
    tp = row['icpt_t_launch']*DAY
    or_period = origin.period
    # tp = random.uniform(0.5 * or_period, 1.5 * or_period)

    rp_vec, vp_vec = origin.theta_to_rv(theta_pe)
    vp_mag = np.linalg.norm(vp_vec)

    min_time = 100
    try:
        insert_dv, rdvz_dv, transfer_orbit, st, et, er = oberth_effect_optimzer(
            ISO,
            rp_vec,
            vp_mag,
            tp,
            min_time,
            max_t,
            optimize_rendezvous=(True),
            period=or_period,
            detect_time=t_detect,
            periods=None
            # tp_window_width=10*YEAR/365
        )
    except:
        print("No Oberth transfer found")
        return

    # ===============================
    # Compute required periapsis direction
    # ===============================
    r_req, v_req = transfer_orbit.theta_to_rv(theta_pe)
    v_req_hat = v_req / np.linalg.norm(v_req)

    # current orbit periapsis direction
    r0, v0 = origin.theta_to_rv(theta_pe)
    v0_hat = v0 / np.linalg.norm(v0)
    #
    # # ===============================
    # # Angle between directions
    # # ===============================
    cos_dtheta = np.clip(np.dot(v0_hat, v_req_hat), -1, 1)
    delta = m.acos(cos_dtheta)
    total_dv = insert_dv

    total_dv += rdvz_dv
    #
    # # ===============================
    # # Apoapsis velocity
    # # ===============================
    # r_a = origin.apoapsis
    # a = origin.a
    # mu = origin.sgp
    #
    # v_ap = m.sqrt(mu * (2 / r_a - 1 / a))
    #
    # # ===============================
    # # Plane change Δv at apoapsis
    # # ===============================
    # dv_plane = 2 * v_ap * m.sin(delta / 2)
    #
    # # ===============================
    # # Total insertion Δv
    # # ===============================
    # insert_dv += dv_plane
    # ===============================
    # Reconstruct rotated orbit (after apoapsis burn)
    # ===============================

    # rotation axis from current to required direction
    axis = np.cross(v0_hat, v_req_hat)
    norm = np.linalg.norm(axis)

    if norm < 1e-10:
        axis = np.array([0, 0, 1])  # fallback
    else:
        axis /= norm

    def rotate(vec):
        return (
                vec * m.cos(delta) +
                np.cross(axis, vec) * m.sin(delta) +
                axis * np.dot(axis, vec) * (1 - m.cos(delta))
        )

    # get apoapsis state
    theta_ap = m.pi
    t_ap = origin.theta_to_time(theta_ap)
    r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

    # rotate state
    r_rot = rotate(r_ap)
    v_rot = rotate(v_ap_vec)
    inc_dv = np.linalg.norm((v_rot - v_ap_vec))
    total_dv += inc_dv

    total_dv = round(total_dv)

    print("Transfer a: ", transfer_orbit.a)
    print("Transfer e: ", transfer_orbit.e)
    print()

    # ==== plotting ====
    # fig = plt.figure()

    # plot original orbit
    plot_orbit(ax, origin, time=t_detect, ThreeDee=True, label="Original")
    # rebuild orbit
    try:
        origin_rot = orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
        # origin_rot.link_time_and_theta(theta_ap, t_ap)
        # origin_rot.normalize()
        # plot rotated orbit
        plot_orbit(ax, origin_rot, time=t_detect, ThreeDee=True, label="Rotated")
    except Exception as e:
        print()
        print("Rotated orbit rebuilding failed: ", e)
        print("Delta V: ", total_dv)

    try:
        plot_orbit(ax, transfer_orbit, time=et, ThreeDee=True, label="Transfer", max_alt=(150 * AU))
    except:
        pass  # lambert sometimes fails

    # plot ISO, earth and jupiter orbit for context
    plot_orbit(ax, ISO, time=et, ThreeDee=True, label="ISO", max_alt=(150 * AU))
    plot_orbit(ax, Earth, time=t_detect, ThreeDee=True, label="Earth")
    plot_orbit(ax, Jupiter, time=t_detect, ThreeDee=True, label="Jupiter")


    print(
                f"ΔV inclination: {inc_dv:.2f} km/s\n"
                f"ΔV insert: {insert_dv:.2f} km/s\n"
                f"ΔV rendezvous: {rdvz_dv:.2f} km/s\n"
                f"ΔV total: {total_dv:.2f} km/s\n"
                f"Intercept distance: {np.linalg.norm(ISO.time_to_rv(et)[0])/AU:.2f} AU\n"
                f"Intercept time: {(et-t_detect)/YEAR:.2f} years\n"
            )



    # printing:
    print(
        f'intercept:\nlaunches: {row["icpt_t_launch"]:.2f} days after detection, arrives {row["icpt_t_arrival"]:.2f} days after detection at a distance of {row["icpt_r"]} AU')
    print(
        f"initial delta v cost is: {row["icpt_idv"]:.2f} km/s, and relative velocity at intercept is {row['icpt_rdv']} km/s\n")

    print(
        f'Rendezvous:\nlaunches: {row["rdvz_t_launch"]:.2f} days after detection, arrives {row["rdvz_t_arrival"]:.2f} days after detection at a distance of {row["rdvz_r"]} AU')
    print(
        f"initial delta v cost is: {row["rdvz_idv"]:.2f} km/s, and relative velocity at intercept is {row['rdvz_rdv']} km/s, for a total delta v of {row["rdvz_total"]:.2f} km/s")


def run_in_background():
    '''run forever generating new datapoints'''
    while True:
        df = get_data(1)
        print("Current # of rows:")
        print(len(df))
        print('---------\n')


def find_optimum_lon_per():

    # ==== settings =====
    icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    if rdvz:
        weight = rdvz_weights
    else:
        weight = icpt_weights

    max_time = 40*YEAR

    lon_vals = np.linspace(0, 360, 10)
    # lon_vals = np.array(([30]))
    all_hists = []

    for lon_per in lon_vals:
        aphelion = (1/5.4507)*5.4507 * AU # Jupiter aphelion
        solar_radius = 696_340
        perihelion = 5*solar_radius
        semi_major_axis = (aphelion + perihelion) / 2
        eccentricity = (aphelion - perihelion) / (aphelion + perihelion)
        # print("Semi major axis: {:.6f} AU".format(semi_major_axis/AU))
        # print("Eccentricity: {:.6f}".format(eccentricity))
        origin = orbit_from_ephemeris(
            semi_major_axis,
            eccentricity,
            m.radians(1.303),
            m.radians(100.46457166),
            m.radians(lon_per),
            m.radians(100.464),
            SGP_SUN
        )
        # N=2000
        # N=5000
        N=10000
        add_dv_hist(origin, max_time, weight, N, PLOT=PLOT, lon_per=np.radians(lon_per))

        hist = get_dv_hist(weight, lon_per=np.radians(lon_per))
        # distribution_histogram(weight, True, lon_per=np.radians(lon_per))
        # plt.figure()
        all_hists.append(hist)

        # probability_map(weight, lon_per=np.radians(lon_per))

    # Convert degrees radians for polar plot
    theta = np.radians(lon_vals)

    dv_90_MS = []
    dv_90_HSP = []
    dv_90_EL = []

    Ezell_Loeb_avg_per_annum = 5
    Hoover_seligman_payne_per_annum = 14
    Marceta_seligman_per_annum = 35
    years = 10

    EL_N = Ezell_Loeb_avg_per_annum * years
    HSP_N = Hoover_seligman_payne_per_annum * years
    MS_N = Marceta_seligman_per_annum * years

    for lon_per, hist in zip(lon_vals, all_hists):
        dv_req_MS = dv_for_confidence(
            weight,
            MS_N,
            0.9,
            lon_per=np.radians(lon_per)
        )
        dv_90_MS.append(dv_req_MS)

        dv_req_EL = dv_for_confidence(
            weight,
            EL_N,
            0.9,
            lon_per=np.radians(lon_per)
        )
        dv_90_EL.append(dv_req_EL)

        dv_req_HSP = dv_for_confidence(
            weight,
            HSP_N,
            0.9,
            lon_per=np.radians(lon_per)
        )
        dv_90_HSP.append(dv_req_HSP)



    # ==== Polar plot ====
    theta = np.radians(lon_vals)

    plt.figure()
    ax = plt.subplot(111, projection='polar')

    ax.plot(theta, dv_90_MS, marker='o', label="Marčeta–Seligman")
    ax.plot(theta, dv_90_HSP, marker='o', label="Hoover-Seligman")
    ax.plot(theta, dv_90_EL, marker='o', label="Ezell-Loeb")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.set_title("ΔV required for 90% mission success")
    # ax.set_rlabel_position(135)
    ax.set_ylabel("ΔV (km/s)")
    ax.grid(True)
    ax.legend()

    plt.show()

    # ==== Results printout ====
    best_idx_MS = np.nanargmin(dv_90_MS)
    best_idx_EL = np.nanargmin(dv_90_EL)
    best_idx_HSP = np.nanargmin(dv_90_HSP)

    print()
    print()
    print("-------------- Marčeta–Seligman Assumption --------------")
    print("Number of ISOs detected: ", MS_N)
    print("Optimum longitude of perihelion: ",
          f"{lon_vals[best_idx_MS]:.2f} deg")
    print("Required ΔV budget: ",
          f"{dv_90_MS[best_idx_MS]:.2f} km/s")

    print()

    print("-------------- Hoover–Seligman–Payne Assumption --------------")
    print("Number of ISOs detected: ", HSP_N)
    print("Optimum longitude of perihelion: ",
          f"{lon_vals[best_idx_HSP]:.2f} deg")
    print("Required ΔV budget: ",
          f"{dv_90_HSP[best_idx_HSP]:.2f} km/s")
    print()

    print("-------------- Ezell–Loeb Assumption --------------")
    print("Number of ISOs detected: ", EL_N)
    print("Optimum longitude of perihelion: ",
          f"{lon_vals[best_idx_EL]:.2f} deg")
    print("Required ΔV budget: ",
          f"{dv_90_EL[best_idx_EL]:.2f} km/s")

def what():
    PLOT=True
    rdvz=True
    # ==== settings =====
    icpt_weights = {"w_insertion": 1, "w_relv": 0, "w_travel_time": 0, "w_intercept_distance": 0, "w_intercept_time": 0}
    rdvz_weights = {"w_insertion": 1, "w_relv": 1, "w_travel_time": 0, "w_intercept_distance": 0, "w_intercept_time": 0}
    if rdvz:
        weight = rdvz_weights
    else:
        weight = icpt_weights

    max_time = 40 * YEAR
    lon_per = 124.14 + 360
    aphelion = 1 * 5.4507 * AU  # Jupiter aphelion
    solar_radius = 696_340
    perihelion = 10 * solar_radius
    semi_major_axis = (aphelion + perihelion) / 2
    eccentricity = (aphelion - perihelion) / (aphelion + perihelion)
    # print("Semi major axis: {:.6f} AU".format(semi_major_axis/AU))
    # print("Eccentricity: {:.6f}".format(eccentricity))
    origin = orbit_from_ephemeris(
        semi_major_axis,
        eccentricity,
        m.radians(1.303),
        m.radians(100.46457166),
        m.radians(lon_per),
        m.radians(100.464),
        SGP_SUN
    )
    # N=2000
    # N=5000
    N = 10000
    add_dv_hist(origin, max_time, weight, N, PLOT=PLOT, lon_per=np.radians(lon_per))


if __name__ == "__main__":
    # what()

    # find_optimum_lon_per()
    # example of using the functions:

    df = get_data()
    dfb = df[df['magnitude_generation_method'] == 'atlas-borisov']
    dfo = df[df['magnitude_generation_method'] == 'omuamua']

    # print(f'fraction omuamua: {len(dfo) / len(df):.2f}, number omuamua: {len(dfo)}')
    # print(f'fraction borisov: {len(dfb) / len(df):.2f}, number borisov: {len(dfb)}')
    # dv_histogram(True, True, df=dfo)
    # plt.title("Omuamua-like dv distribution")
    # plt.show()
    # dv_histogram(True, True, df=dfb)
    # plt.title("borisov-like dv distribution")
    # plt.show()

    DV_THRESHOLD = 24
    MAX_RDVZ_DISTANCE = 200  # AU

    total_dv = df['rdvz_total']

    df_reach = df[
        total_dv.notna() &
        (total_dv < DV_THRESHOLD) &
        (df['rdvz_r'] < MAX_RDVZ_DISTANCE)
        ]

    total = len(df_reach)

    pct_200 = 100 * np.sum(df_reach['rdvz_r'] < 200) / total
    pct_150 = 100 * np.sum(df_reach['rdvz_r'] < 150) / total
    pct_100 = 100 * np.sum(df_reach['rdvz_r'] < 100) / total

    print(f"Under 200 AU: {pct_200:.2f}%")
    print(f"Under 150 AU: {pct_150:.2f}%")
    print(f"Under 100 AU: {pct_100:.2f}%")
    dv_histogram(False, True, df_reach)
    plt.show()
    probability_map_df(dfb, True)
    plt.show()

    plt.figure()

    plt.hist(
        df_reach['rdvz_r'].dropna(),
        bins=40,
        density=True
    )

    plt.xlabel("Rendezvous distance (AU)")
    plt.ylabel("Probability density")
    plt.title("Reachable ISO rendezvous distance")
    plt.show()

    plt.figure()

    plt.hist(
        df_reach['rdvz_t_arrival'].dropna()/365,
        bins=40,
        density=True
    )

    plt.xlabel("Rendezvous intercept time (Years)")
    plt.ylabel("Probability density")
    plt.title("Reachable ISO rendezvous time")
    plt.show()

    plt.hist(
        df_reach['rdvz_rdv'].dropna(),
        bins=40,
        density=True
    )

    plt.xlabel("Rendezvous velocity (km/s)")
    plt.ylabel("Probability density")
    plt.title("Reachable ISO rendezvous velocities")
    plt.show()

    plt.figure()

    plt.hist(
        df_reach['rdvz_idv'].dropna(),
        bins=40,
        density=True
    )

    plt.xlabel("Insertion + plane-change ΔV (km/s)")
    plt.ylabel("Probability density")
    plt.title("Reachable ISO insertion ΔV")
    plt.show()

    df = df.sort_values('rdvz_total', ignore_index=True)
    print(df)
    if PLOT:
        ax = get_solar_system_ax()
        plot_from_row(ax, df.iloc[0], 20)
        plt.axis('equal')
        plt.legend()
        plt.show()

    run_in_background()
