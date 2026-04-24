from src.get_ISO import get_ISO

from src.orbit import Orbit, plot_orbit, trajectory_optimizer
from src.utilities import AU, SGP_SUN, YEAR, DAY
from src.examples import Earth, Mars, Jupiter, Omuamua, to_epoch
import matplotlib.pyplot as plt
import numpy as np
import datetime


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
ob2 = Omuamua

res = 50
start_range = np.linspace(ob2.t_p-0.5*YEAR, ob2.t_p + YEAR,res)
end_range = np.linspace(ob2.t_p, ob2.t_p + 5*YEAR,res)


dv_arr, idx = Orbit.porkchop_plot2(ob1.time_to_rv,ob2.time_to_rv,start_range,end_range,ob1.sgp, rendezvous=True) # type: ignore

dv1,dv2,st,et,_ = trajectory_optimizer(ob1,ob2,start_range[0],end_range[-1],w_relv=1)

traj_ob = Orbit.orbit_from_lambert(ob1.time_to_rv(st)[0], ob2.time_to_rv(et)[0],st,et,ob1.sgp)

print(f"delta v: {dv_arr[*idx]}, at start time: {start_range[idx[0]]:.2f} and end time: {end_range[idx[1]]}")
print(f"delta v opt: {dv1+dv2}\t at start time:{st:.2f} and end time {et:.2f}")

DAY = 1

plt.imshow(np.array(dv_arr).T, origin='lower', extent=((start_range[0])/DAY,(start_range[-1])/DAY,(end_range[0])/DAY, (end_range[-1])/DAY))
plt.scatter((start_range[idx[0]])/DAY,(end_range[idx[1]])/DAY,color="blue")
plt.scatter((st)/DAY,(et)/DAY,color="red")

plt.xlabel("Departure (days)")
plt.ylabel("Arrival (days)")

plt.show()
ax = plt.figure().add_subplot(projection='3d')
plot_orbit(ax,ob1,st)
plot_orbit(ax,ob2,et, max_alt=10*AU)
plot_orbit(ax,traj_ob, et, trail=(abs(traj_ob.time_to_theta(st)-traj_ob.time_to_theta(et))))
plt.axis('scaled')
plt.show()
