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

# ==== settings =====

T = 0 # time window of ISO generation
# interecept only optimizes for minimum insertion, rendezvous optimises for minimum total dV
icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
weight = icpt_weights

detect_distance = 3*AU
max_time = 10*YEAR
origin = Earth


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
