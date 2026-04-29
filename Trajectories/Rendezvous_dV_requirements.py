'''
Script for generating a distributions of dV expected for the ISO intercept


'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.orbit import Orbit, trajectory_optimizer
from src.get_ISO import get_ISO
from src.examples import Earth
from src.utilities import AU, YEAR
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
from pathlib import Path

icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}


# === probability functions =====

def mission_success_probability(detection_distance:float, dV_budget:int, N:int, weight:dict)->float:
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
    p_ISO = ISO_probability(detection_distance,dV_budget, weight)
    p_least_one = 1-(1-p_ISO)**N
    return p_least_one

    
def ISO_probability(rm:float,dV_budget:int, weight)->float:
    '''calculate individual chance of success for given dv budget and detection distance,
    currently only works with integer dV budgets

    :param detection_distance: distance (in AU) from the sun that ISOs are detected
    :type: float
    :param dV_buget: total mission dV budget
    :type: int
    :param weight: optimizer weight, i.e. an intercept or rendezvouz
    :type: dict'''
    hist = get_dv_hist(rm,weight)
    return np.sum(hist[:m.floor(dV_budget)+1])

# ==== histogram generation ====

def add_dv_hist(rm, weights, N)->None:
    '''Adds to the dv histogram for the different weights,
    rm is detection distance (and also simulation distance, such that every generated ISO is detected)'''
    np.seterr(all="ignore")
    path = Path(__file__).parent.parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            count = int(lines[0])
            hist = [int(x) for x in lines[1:]]
    except:
        hist = [0 for _ in range(100)]
        count = 0

    ISOs = get_ISO() # sample of ISOs
    while len(ISOs) < N:
        ISOs.extend(get_ISO())
    count += len(ISOs)

    for ISO in tqdm(ISOs,desc="studying ISOs"):
        detect_theta = ISO.crosses_altitude(rm*AU)
        if detect_theta is None: continue
        detect_time = ISO.theta_to_time(-detect_theta)
        try:
            insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(Earth,ISO,detect_time,detect_time+10*YEAR, **weights)
        except (ArithmeticError,ValueError): continue
        dv = round(insert_dv*weights["w_insertion"] + rdvz_dv*weights["w_relv"])
        if dv >= 100: continue
        hist[dv] += 1
    
    # Save
    path = Path(__file__).parent.parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    with open(path, "w") as file:
        file.write(str(count) + '\n')
        file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(rm, weights)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    path = Path(__file__).parent.parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"

    with open(path, "r") as file:
        lines = file.readlines()
        count = int(lines[0])
        hist = [int(x) for x in lines[1:]]
    return [x/count for x in hist]


def probability_map(rm:float, weight:dict, guesses:bool = True, show:bool=True):
    '''generate probability map of N over dV,'''

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
    PP = np.vectorize(mission_success_probability)(rm,NN,VV,weight)
    plt.imshow(PP,origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    plt.colorbar(location="right", label="Probability of mission success")
    CS = plt.contour(PP,levels=[0.9],origin="lower",aspect="auto", extent=(N_range[0],N_range[-1],V_range[0],V_range[-1]))
    plt.clabel(CS, fmt=lambda x: f"{x*100:.0f}%")
    plt.ylabel('Delta V budget (km/s)')
    plt.xlabel('number of ISOs during mission time')
    plt.title(f"Probability map for {"rendezvous" if weight["w_relv"] else "intercept"} with detection range of {rm} AU\nAnd estimated ISO detections during {years} year mission")
    if guesses:
        plt.plot([EL_N,EL_N],[5,48], ls='--', color="gray")
        plt.text(EL_N+1, 40, "Ezell, Loeb mean", color="gray")
        plt.plot([HSP_N,HSP_N],[5,48], ls='--', color="gray")
        plt.text(HSP_N+1, 30, "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.plot([MS_N,MS_N],[5,48], ls='--', color="gray")
        plt.text(MS_N-1, 20, "Marčeta, Seligman mean", ha="right", color="gray")
    if show:
        plt.show()

def distribution_histogram(rm:float, weight:dict,  show:bool=True):
    '''generate the histogram of the dV requirements'''

    hist = get_dv_hist(rm, weight)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.title(f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if weight["w_relv"] else "intercept"}\nwith detection range {rm} AU. (Normalization includes unreachable ISOs)")
    if show:
        plt.show()


if __name__ == "__main__":

    

    # ==== settings =====    
    weight = icpt_weights
    detect_distance = 3
    # add_dv_hist(5, weight,0)
    # probability_map(2, icpt_weights, False, False)
    # plt.figure()
    # probability_map(3, icpt_weights, False, False)
    # plt.figure()
    # probability_map(5, icpt_weights, False, False)
    # plt.figure()
    # probability_map(2, rdvz_weights, False, False)
    # plt.figure()
    # probability_map(3, rdvz_weights, False, False)
    # plt.figure()
    # probability_map(5, rdvz_weights, False, False)
    # plt.show()


    # distribution_histogram(2, icpt_weights, False)
    # plt.figure()
    # distribution_histogram(3, icpt_weights, False)
    # plt.figure()
    # distribution_histogram(5, icpt_weights, False)
    # plt.figure()
    # distribution_histogram(2, rdvz_weights, False)
    # plt.figure()
    # distribution_histogram(3, rdvz_weights, False)
    # plt.figure()
    # distribution_histogram(5, rdvz_weights, False)
    # plt.show()
    
    probability_map(3,icpt_weights)

    

    while True:
        add_dv_hist(2, rdvz_weights, 0)
        add_dv_hist(3, rdvz_weights, 0)
        add_dv_hist(5, rdvz_weights, 0)

    # add_dv_hist(5,weight, 2000)
    # probability_map(5, weight)




    # hist = get_dv_hist(3, weight)
    # print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    # plt.bar(range(100),hist,width=1)
    # plt.xlabel("dV requirement")
    # plt.ylabel("probability density")
    # plt.show()
