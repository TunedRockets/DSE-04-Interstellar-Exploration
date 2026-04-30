''' 
file for optimizing the optimizer
'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.orbit import Orbit, trajectory_optimizer, porkchop_from_orbits
from src.examples import Earth, Jupiter, Mars, Omuamua
from src.utilities import AU, YEAR, inside_modulo_bounds
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math as m
from tqdm import tqdm




origin = Earth
destination = Omuamua
destination.t_p = 5*YEAR
resolution = 100

start_range = np.linspace(0,10*YEAR,resolution)
end_range = np.linspace(0,10*YEAR,resolution)
xx,yy = np.meshgrid(start_range,end_range)
dv_arr, idx = porkchop_from_orbits(origin,destination,start_range,end_range) # type: ignore
print("porkchop done")
idv,rdv, s, e,_ = trajectory_optimizer(origin,destination,start_range[0], end_range[-1], w_relv=1, w_intercept_time=0.005)
print(f"optimizer: {idv+rdv}")
print(f"porkchop min: {dv_arr[idx]}")




plt.imshow(dv_arr.T, origin="lower", vmax=60, extent=(start_range[0], start_range[-1], end_range[0], end_range[-1]))
plt.scatter(start_range[idx[0]], end_range[idx[1]], color="blue")
plt.scatter(s, e, color="red")
plt.show()


