''' 
file for optimizing the optimizer
'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.orbit import Orbit, trajectory_optimizer, porkchop_from_orbits
from src.examples import Earth, Jupiter, Mars
from src.utilities import AU, YEAR, inside_modulo_bounds
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm




origin = Earth
destination = Mars
resolution = 100

start_range = np.linspace(0,3*YEAR,resolution)
end_range = np.linspace(0,3*YEAR,resolution)

dv_arr, idx = porkchop_from_orbits(origin,destination,start_range,end_range) # type: ignore
print("porkchop done")

# points of interest:
node_start = [origin.theta_to_time(x) for x in [0,m.pi, -origin.arg_p, m.pi - origin.arg_p]] #
node_end = [destination.theta_to_time(x) for x in [0,m.pi, -destination.arg_p, m.pi - destination.arg_p]]
start_time = start_range[0]
end_time = end_range[-1]
poi_start = []
poi_end = []
for n in node_start:
    poi_start.extend(inside_modulo_bounds(start_time, n, end_time, origin.period))
for n in node_end:
    poi_end.extend(inside_modulo_bounds(start_time, n, end_time, destination.period))

# hohmann:
if origin.e < 1 and destination.e < 1:
    t = origin.hohmann_time(destination)
    poi_start.extend(inside_modulo_bounds(start_time,t,end_time, origin.synodic_period(destination)))
    t += origin.hohmann_travel_time(destination)
    poi_end.extend(inside_modulo_bounds(start_time,t,end_time, origin.synodic_period(destination)))


xx, yy = np.meshgrid(poi_start,poi_end)

plt.imshow(dv_arr.T, origin="lower", vmax=15, extent=(start_range[0], start_range[-1], end_range[0], end_range[-1]))
plt.scatter(xx,yy)
plt.show()


