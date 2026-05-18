

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.orbit import Orbit
from src.utilities import AU
import numpy as np
from Rendezvous_dV_requirements import get_data, recreate_ISO
from tqdm import tqdm
import matplotlib.pyplot as plt

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
a_sigma = np.average(a_arr) * AU
i_sigma = np.average(i_arr) * np.pi/180
raan_sigma = np.average(raan_arr) * np.pi/180
argp_sigma = np.average(argp_arr) * np.pi/180
tp_sigma = np.average(tp_arr)


undetection_distance = 6 * AU # not relevant for now (can be releavant if turned to rv)



def orbit_shuffle(ob:Orbit)->None:
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
    ob.RAAN += draan
    ob.arg_p += dargp
    ob.t_p += dtp


df = get_data()
df = df[df["rdvz_total"] < 19.3]

errors = []

for i, row in tqdm(df.iterrows(), desc="making disturbances"):
    ob,_,_ = recreate_ISO(row)
    t_end = row['rdvz_t_arrival'] - row['time_until_periapsis'] + row['t_p'] # time of arrival
    r = ob.time_to_rv(t_end)[0]
    orbit_shuffle(ob)
    r_err = ob.time_to_rv(t_end)[0]
    errors.append(r_err - r)

errors = np.array(errors)
dists = np.linalg.norm(errors, axis=1)
print(f'avg={np.average(dists)}\tstd={np.std(dists)}\tmax={np.max(dists)}\tmin={np.min(dists)}')

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(errors[:,0],errors[:,1],errors[:,2],color="blue") # type:ignore
ax.scatter(0,0,0, lw=3, color="red")
plt.show()
