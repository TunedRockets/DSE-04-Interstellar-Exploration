'''
Script for generating a distributions of dV expected for the ISO intercept


'''

from src.orbit import Orbit, trajectory_optimizer
from src.get_ISO import get_ISO
from src.examples import Earth
from src.utilities import AU, YEAR
import numpy as np
from tqdm import tqdm

# ==== settings =====

T = 0 # time window of ISO generation
icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
detect_distance = 3*AU
max_time = 10*YEAR
# ==== generate scenario ======
ISOs = get_ISO(T) # sample of ISOs


# === inspect ISOs for their properties ====
ISO_stats = []

for ISO in tqdm(ISOs,desc="Studying each ISO"):
    dete
    opt_icpt = trajectory_optimizer()

