'''
Script for generating a distributions of dV expected for the ISO intercept


'''

from src.orbit import Orbit, trajectory_optimizer
from src.get_ISO import get_ISO
from src.examples import Earth
from src.utilities import AU, YEAR
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
from pathlib import Path





def add_dv_hist(rm, weights, N)->None:
    '''Adds to the dv histogram for the different weights'''
    np.seterr(all="ignore")
    path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    
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
            insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(origin,ISO,detect_time,detect_time+max_time, **weights)
        except: continue
        insert_dv = round(insert_dv)
        if insert_dv > 100: continue
        hist[insert_dv] += 1
    
    # Save
    path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    with open(path, "w") as file:
        file.write(str(count) + '\n')
        file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(rm, weights)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    

    with open(path, "r") as file:
        lines = file.readlines()
        count = int(lines[0])
        hist = [int(x) for x in lines[1:]]
    return [x/count for x in hist]






if __name__ == "__main__":

    

    # ==== settings =====

    T = 0 # time window of ISO generation
    # interecept only optimizes for minimum insertion, rendezvous optimises for minimum total dV
    icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    weight = icpt_weights

    detect_distance = 3*AU
    max_time = 10*YEAR
    origin = Earth

    
    rm = 4
    add_dv_hist(rm,weight,2000)

    hist = get_dv_hist(rm, weight)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.show()
    raise NotImplementedError()
    


    # ==== generate scenario ======
    ISOs = get_ISO() # sample of ISOs
    while len(ISOs) < 400:
        ISOs.extend(get_ISO())
    print(f"Number of generated ISOs: {len(ISOs)}")


    # === inspect ISOs for their properties ====
    ISO_stats = []
    invalid_count = 0
    for ISO in tqdm(ISOs,desc="Studying each ISO"):
        detect_theta = ISO.crosses_altitude(detect_distance)
        if detect_theta is None: continue
        detect_time = ISO.theta_to_time(-detect_theta)
        try:
            insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(origin,ISO,detect_time,detect_time+max_time, **weight)
        except: 
            invalid_count += 1 # to be handled later
            continue

        stats = {
            "insertion dV": insert_dv,
            "rendezvous dV": rdvz_dv,
            "total dV": insert_dv + rdvz_dv,
            "launch time": st,
            "intercept time":et,
            "intercept distance":er 
        }
        ISO_stats.append(stats)

    # === plot data ===

    idVs = [x["insertion dV"] for x in ISO_stats]
    rdVs = [x["rendezvous dV"] for x in ISO_stats]


    plt.hist(idVs,bins=range(m.floor(min(idVs)), m.ceil(max(idVs))+1, 3),density=True)
    plt.xlabel("Insertion dV (km/s)")
    plt.ylabel("density")
    plt.show()

    plt.hist(rdVs,bins=range(m.floor(min(rdVs)), m.ceil(max(rdVs))+1, 3),density=True)
    plt.xlabel("relative velocity at intercept (km/s)")
    plt.ylabel("density")
    plt.show()
