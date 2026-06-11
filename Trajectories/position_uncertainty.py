

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import jkat
import numpy as np
import math as m
from Rendezvous_dV_requirements import get_data, recreate_ISO
from contingency_analysis import simple_ISO_and_trans
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy

# interstellar uncertainties from the JPL SBDL (1-sigma)
e_arr = [2.469e-5, 2.1064e-5, 1.9039e-5]
a_arr = [6.9616e-7, 1.0015e-4, 5.8802e-6] # [AU]
# q_arr = [3.7743e-6,] # [deg]
i_arr = [1.9023e-5, 2.8826e-4, 1.6673e-5] # [deg]
raan_arr = [7.059e-5, 2.5422e-4, 4.5021e-5] # [deg]
argp_arr = [1.0119e-4, 1.2495e-3, 1.1012e-4] # [deg]
tp_arr = [5.0807e-5, 2.6424e-4, 1.4206e-4] #[s]

# turn into sigmas:
e_sigma = np.average(e_arr)
a_sigma = np.average(a_arr) * jkat.AU
i_sigma = np.average(i_arr) * np.pi/180
raan_sigma = np.average(raan_arr) * np.pi/180
argp_sigma = np.average(argp_arr) * np.pi/180
tp_sigma = np.average(tp_arr)


undetection_distance = 6 * jkat.AU # not relevant for now (can be releavant if turned to rv)



def orbit_shuffle(ob:jkat.Orbit)->None:
    '''shuffle orbit based on the uncertainty values,
    changes in place(!)'''
    de = np.random.randn() * e_sigma
    da = np.random.randn() * a_sigma
    di = np.random.randn() * i_sigma
    draan = np.random.randn() * raan_sigma
    dargp = np.random.randn() * argp_sigma
    dtp = np.random.randn() * tp_sigma
    ob.e += de
    ob.a += da
    ob.i += di
    ob.raan += draan
    ob.argp += dargp
    ob.tp += dtp

def error_distance():
    ob_list = simple_ISO_and_trans()

    errors = []
    distances = []

    for ISO, trans, ts, te in tqdm(ob_list, desc="making disturbances"):
        for _ in range(10):
            ISO2 = copy(ISO) # since shuffling is in place we need a copy
            r = ISO2.t2rvec(te)
            distances.append(np.linalg.norm(r))
            orbit_shuffle(ISO2)
            r_err = ISO2.t2rvec(te)
            errors.append(r_err - r)
    errors = np.array(errors)
    dists = np.linalg.norm(errors, axis=1)
    print(f'avg={np.average(dists)}\tstd={np.std(dists)}\tmax={np.max(dists)}\tmin={np.min(dists)}')
    distances = np.array(distances)/jkat.AU
    print(f'avg={np.average(distances)}\tstd={np.std(distances)}\tmax={np.max(distances)}\tmin={np.min(distances)}')

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(errors[:,0],errors[:,1],errors[:,2],color="blue") # type:ignore
    ax.scatter(0,0,0, lw=3, color="red")
    plt.show()



# figure out detection distance:
def HG(H:float, r_delta:float, r_obj:float, phi:float):
    # HG constants:
    A1 = 3.332
    A2 = 1.862
    B1 = 0.631
    B2 = 1.218
    G = 0.15
    varphi1 = m.exp(-A1 * m.tan(phi/2)**B1)
    varphi2 = m.exp(-A2 * m.tan(phi/2)**B2)
    phase = 2.5*m.log10((1-G)*varphi1 + G*varphi2)

    V = H + 5*m.log10(r_delta) + 5*m.log10(r_obj) - phase
    return V


error_distance()


# # LORRI number:
# r_delta = 44 # [AU]
# r_obj = 1_900_000 / AU # [AU]
# phi = 0 # roughly

# V = HG(10.4, r_delta,r_obj,phi)
# print(V)

# # Our worst case:
# H = 12
# r_delta = 50 # [au]
# phi = 0
# V = 9.5
# F = lambda r: HG(H, r_delta, r/AU, phi)

# r = root_finder_bisection(F,1,1_000_000)
# print(f"{r=:.1f}\t for: {r_delta=:.1f} AU")

