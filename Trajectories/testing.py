

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.get_ISO import get_ISO
from src.orbit import Orbit, plot_orbit, trajectory_optimizer, porkchop_plot
from src.utilities import AU, SGP_SUN, YEAR, DAY
from src.examples import Earth, Mars, Jupiter, Omuamua, to_epoch
import matplotlib.pyplot as plt
import numpy as np
import datetime



ob1 = Earth
ob2 = Jupiter

start_time = ob2.t_p
end_time = start_time + 3*YEAR
range_time = np.linspace(start_time,end_time,100)


# dv_arr, idx = Orbit.porkchop_intercept(ob1,ob2,range_time,range_time) # type: ignore
dv_arr, idx = porkchop_plot(ob1.time_to_rv,ob2.time_to_rv,range_time,range_time,ob1.sgp) # type: ignore

dv1,dv2,st,et,_ = trajectory_optimizer(ob1,ob2,start_time,end_time,w_relv=0)


print(f"delta v: {dv_arr[*idx]}, at idx: {idx}")
print(f"delta v opt: {dv1 + dv2}")



plt.imshow(np.array(dv_arr).T, origin='lower', vmax=30, extent=(0,(end_time-start_time)/DAY,0,(end_time-start_time)/DAY))
plt.scatter((range_time[idx[0]]-start_time)/DAY,(range_time[idx[1]]-start_time)/DAY,color="blue")
plt.scatter((st-start_time)/DAY,(et-start_time)/DAY,color="red")
# plt.gca().invert_yaxis()
plt.xlabel("Departure (days)")
plt.ylabel("Arrival (days)")
plt.show()