from src.get_ISO import get_ISO

from src.orbit import Orbit, plot_orbit, trajectory_optimizer
from src.utilities import AU, SGP_SUN, YEAR, DAY
from src.examples import Earth, Mars, Jupiter, Omuamua
import matplotlib.pyplot as plt
import numpy as np


# obs = get_ISO()
# # for ob in obs:
# #     print(ob)
# print(len(obs))

# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(0,0,0,color='red', lw=3)

# LIMIT = 6*AU
# # plot:
# for ob in obs[:50]:
#     lim = ob.crosses_altitude(LIMIT)
#     if lim is None: continue
#     if np.linalg.norm(ob.time_to_rv(0)[0]) > LIMIT: continue

#     plot_orbit(ax,ob, max_alt=LIMIT)

# plot_orbit(ax,Earth, color='Blue')
# plot_orbit(ax,Mars,color='Red')
# plot_orbit(ax,Jupiter,color='darkorange')

# plt.axis('scaled')
# plt.show()




ob1 = Earth
ob2 = Jupiter

start_time = ob2.t_p
end_time = start_time + 2*YEAR
range_time = np.linspace(start_time,end_time,100)



# dv_arr, idx = Orbit.porkchop_intercept(ob1,ob2,range_time,range_time) # type: ignore
dv_arr, idx = Orbit.porkchop_plot2(ob1.time_to_rv,ob2.time_to_rv,range_time,range_time,ob1.sgp) # type: ignore

dv1,dv2,st,et,_ = trajectory_optimizer(ob1,ob2,start_time,end_time,w_relv=1)


print(f"delta v: {dv_arr[*idx]}, at idx: {idx}")
print(f"delta v opt: {dv1 + dv2}")



plt.imshow(np.array(dv_arr).T, origin='lower', vmax=30, extent=(0,(end_time-start_time)/DAY,0,(end_time-start_time)/DAY))
plt.scatter((range_time[idx[0]]-start_time)/DAY,(range_time[idx[1]]-start_time)/DAY,color="blue")
plt.scatter((st-start_time)/DAY,(et-start_time)/DAY,color="red")
# plt.gca().invert_yaxis()
plt.xlabel("Departure (days)")
plt.ylabel("Arrival (days)")

plt.show()
