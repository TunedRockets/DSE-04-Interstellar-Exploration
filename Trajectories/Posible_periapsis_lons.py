import jkat as jk
import numpy as np
from jkat.utils import longp
import math as m

from Structures.holistic_mass_solver import dV_rdvz
from Trajectories.Rendezvous_dV_requirements import MAX_MISSION_TIME, LONGP_NUM
from src.helio_optim import *
from Rendezvous_dV_requirements import get_parking
import os

import pandas as pd
from pathlib import Path

from src.get_ISO import get_ISO
from functools import partial
from tqdm import tqdm
import multiprocessing as mp


AU = jk.AU
DAY = jk.DAY
YEAR = jk.YEAR

dV_inclination_b = 3.500
dV_oberth_b = 4.000
dV_rendezvous_b = 13.000
dv_budget = (dV_inclination_b, dV_oberth_b, dV_rendezvous_b)
longp_num = 50


PATH_TO_DATA = Path(__file__).parent.parent / "data"
PICKLE_NAME = "Possible_ISOs_with_Budget"

def budget_suffix(dv_budget: tuple[float, float, float]) -> str:
    return f"_inc{dv_budget[0]:.3f}_ob{dv_budget[1]:.3f}_ren{dv_budget[2]:.3f}"

PICKLE_NAME = PICKLE_NAME + budget_suffix(dv_budget)

USER_NAME = os.getlogin()

# TODO: MAKE IT TAKE A LIST OF LONG PS FOR THE WINDOW STUFF

def job(ISOtuple, longps, dv_budget):

    dv_inc, dv_oberth, dv_rendezvous = dv_budget
    ISO, detect_t, g_type = ISOtuple

    np.seterr(all="ignore")

    detect_r = ISO.r(ISO.f(detect_t)) / AU

    rows = []


    for longp in longps:

        try:
            possible, res = check_if_possible(
                dv_inc,
                dv_oberth,
                dv_rendezvous,
                get_parking(longp),
                ISO,
                ISO.tp + MAX_MISSION_TIME * YEAR,
                detect_t
            )

            rows.append({
                "detection_r": detect_r,
                "periapsis": ISO.periapsis / AU,
                "magnitude_generation_method": g_type,
                "time_until_periapsis": (ISO.tp - detect_t) / DAY,
                "parameter": ISO.p,
                "e": ISO.e,
                "i": ISO.i,
                "RAAN": ISO.raan,
                "arg_p": ISO.argp,
                "t_p": ISO.tp,
                "ISO_excess_velocity": ISO.vinf,

                "longp": longp,

                "h_tdv": res["dv0"],
                "h_idv": res["dv1"],
                "h_rdv": res["dv2"],
                "h_possible": possible,
            })

        except (ArithmeticError, ValueError, AssertionError) as e:
            # print("LMAO  ERROR : ", e)
            rows.append({
                "detection_r": detect_r,
                "periapsis": ISO.periapsis / AU,
                "magnitude_generation_method": g_type,
                "time_until_periapsis": (ISO.tp - detect_t) / DAY,
                "parameter": ISO.p,
                "e": ISO.e,
                "i": ISO.i,
                "RAAN": ISO.raan,
                "arg_p": ISO.argp,
                "t_p": ISO.tp,
                "ISO_excess_velocity": ISO.vinf,

                "longp": longp,

                "h_tdv": np.inf,
                "h_idv": np.inf,
                "h_rdv": np.inf,
                "h_possible": False,
            })
            pass

    if len(rows) == 0:
        return [{
            "h_possible": False,
            "longp": np.nan,
            "h_tdv": np.nan,
            "h_idv": np.nan,
            "h_rdv": np.nan,
            "detection_r": detect_r,
            "ISO_excess_velocity": ISO.vinf
        }]
    else:
        return rows


def study_batch_multi(dv_budget, longps, gen_type=''):

    ISOs = get_ISO()
    F = partial(job, longps=longps, dv_budget=dv_budget)

    with mp.Pool() as p:
        res = tqdm(
            p.imap_unordered(F, ISOs),
            desc=f"Studying ISOs",
            total=len(ISOs)
        )

        resl = list(res)

    print("Pool closed")

    flat = [row for sub in resl for row in sub]

    return pd.DataFrame(flat)


def get_data(longps, extra_batches: int = 0, gen_type: str = "") -> pd.DataFrame:
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
            data: pd.DataFrame = pd.read_pickle(PATH_TO_DATA / (PICKLE_NAME + USER_NAME))
        except (FileNotFoundError):
            data = pd.DataFrame()
        new = [data]
        for i in range(extra_batches):
            print('============================================')
            print(f"Generating batch {i + 1} of {extra_batches}:")
            print('============================================')
            new.append(
                study_batch_multi(
                    dv_budget=dv_budget,
                    gen_type=gen_type,
                    longps=longps
                )
            )
        data = pd.concat(new, ignore_index=True)
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

import matplotlib.pyplot as plt


def plot_reachability_vs_longitude(df, threshold_pct=0.76/100):

    grouped = df.groupby("longp")["h_possible"].mean() * 100

    angles = grouped.index.values
    values = grouped.values

    # close the loop for polar plot
    angles = np.append(angles, angles[0])
    values = np.append(values, values[0])

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="polar")

    ax.plot(angles, values, marker="o")
    ax.fill(angles, values, alpha=0.2)

    ax.axhline(threshold_pct * 100, linestyle="--")

    ax.set_title("Reachability vs Parking Longitude")
    plt.show()
def run_in_background():
    '''run forever generating new datapoints'''
    while True:
        df = get_data(1)
        print("Current # of rows:")
        print(len(df))
        print('---------\n')


def _test_check_if_possible():
    ISO,detect_t ,_ = get_ISO()[0]
    longp = 0.0

    dv_budget = (3.0, 3.0, 3.0) #
    jk.add_solar_system()
    jk.plot(get_parking(longp))
    plt.show()

    possible, res = check_if_possible(
        dv_budget[0],
        dv_budget[1],
        dv_budget[2],
        get_parking(longp),
        ISO,
        ISO.tp + MAX_MISSION_TIME * YEAR,
        detect_t
    )
    # print(possible)
    # print(res)
if __name__ == "__main__":

    # _test_check_if_possible()
    longps = np.linspace(-np.pi, np.pi, longp_num)
    df = get_data(longps, extra_batches=0)
    print()
    print("Full data frame: ")
    print()
    print(df)
    print()
    dfpos = df[df['h_possible']]
    print()
    print("Possible data frame: ")
    print()
    print(dfpos)
    print()
    plot_reachability_vs_longitude(df)
    # run_in_background()