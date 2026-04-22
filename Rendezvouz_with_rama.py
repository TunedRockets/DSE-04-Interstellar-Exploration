'''
Proof of concept code for calculating ISO interceptor trajectories
using the Kepler orbit script
'''
import src.orbit as ob
from src.utilities import *
import matplotlib.pyplot as plt
import numpy as np

# ====== earth ==========

AU = 1.5e8 # [km] 
SUN_SGP = 1.327e11 # [km^3/s^2]
YEAR = 60*60*24*365 # [s]

Earth = ob.Orbit(AU,0.05,0,0,0,0,SUN_SGP)

# ==== Rama =======
Rama_e = 1 + np.random.rand() # between 1-2
Rama_pe = (0.8 + 2*np.random.rand())*AU
Rama_i = np.pi*np.random.rand()*0.2 # assume ecliptic for now
Rama_RAAN = 2*np.pi*np.random.rand() # since i is 0
Rama_arg_p = 2*np.pi*np.random.rand() / 2
Rama_t_p = 0 # use t_p as reference


Rama = ob.Orbit(Rama_pe, Rama_e, Rama_i, Rama_RAAN, Rama_arg_p, Rama_t_p, SUN_SGP)
Rama.periapsis = Rama_pe # since constructor uses parameter not periapsis

rama_below_limit:float = Rama.crosses_altitude(AU*3) # type:ignore

start_time = Rama.theta_to_time(-rama_below_limit)
end_time = Rama.theta_to_time(rama_below_limit)*1

# ==== intercept =====
rdvz_start = list(np.linspace(start_time,0,50))
rdvz_end = list(np.linspace(0,end_time, 50))

def f1(time): return Earth.time_to_rv(time)
def f2(time): return Rama.time_to_rv(time)

dv_arr, idx = ob.Orbit.porkchop_plot(f1,f2,rdvz_start,rdvz_end,Earth.sgp, rendezvous=False)
rdvz_time_start = rdvz_start[idx[0]]
rdvz_time_end = rdvz_end[idx[1]]
r_start = f1(rdvz_time_start)[0]
r_end = f2(rdvz_time_end)[0]
print(f"{r_start=}\t{r_end=}")
Rdvz = ob.Orbit.orbit_from_lambert(r_start,r_end,rdvz_time_start,rdvz_time_end,Earth.sgp)



# debug plot
plt.scatter(r_start[0],r_start[1],color="red")
plt.scatter(r_end[0],r_end[1],color="red")


# # check:
# assert np.linalg.norm(Rdvz.time_to_rv(rdvz_time_start)[0] - r_start) < 10
# assert np.linalg.norm(Rdvz.time_to_rv(rdvz_time_end)[0] - r_end) < 10

rdvz_theta_start = Rdvz.time_to_theta(rdvz_time_start)
rdvz_theta_end = Rdvz.time_to_theta(rdvz_time_end)
dv_at_rdvz = np.linalg.norm(Rama.time_to_rv(rdvz_time_end)[1] - Rdvz.time_to_rv(rdvz_time_end)[1])
dv_required = dv_arr[*idx]


# # ==== dv plot =====

plt.scatter(rdvz_start[idx[0]], rdvz_end[idx[1]])
plt.xlabel("start time [s]")
plt.ylabel("end time [s]")
plt.imshow(dv_arr)
plt.colorbar()
plt.show()



# ==== ob plot config ====
ax = plt.figure().add_subplot(projection='3d')

# draw sun:
ax.scatter(0,0,0,color='red', lw=3)


# draw earth:

earth_ob = Earth.point_locus()
earth_pe = Earth.theta_to_rv(0)[0]


ax.plot(earth_ob[:,0],earth_ob[:,1],earth_ob[:,2], color = "blue")
ax.scatter(earth_ob[-1,0],earth_ob[-1,1], earth_ob[-1,2], color = "blue")
ax.scatter(earth_pe[0],earth_pe[1],earth_pe[2], color = "blue", marker="v")

# draw rama:
s_theta = Rama.time_to_theta(start_time)
e_theta = Rama.time_to_theta(end_time)
rama_ob = Rama.point_locus(s_theta,e_theta)
rama_pe = Rama.theta_to_rv(0)[0]

ax.plot(rama_ob[:,0],rama_ob[:,1],rama_ob[:,2], color = "green")
ax.scatter(rama_ob[-1,0],rama_ob[-1,1], rama_ob[-1,2], color = "green")
ax.scatter(rama_pe[0],rama_pe[1], rama_pe[2], color = "green", marker="v")

# # draw Rdvz:
earth_takeoff = Earth.time_to_rv(rdvz_time_start)[0]
rama_landing = Rama.time_to_rv(rdvz_time_end)[0]
rdvz_ob = Rdvz.point_locus(rdvz_theta_start,rdvz_theta_end)
ax.plot(rdvz_ob[:,0],rdvz_ob[:,1], rdvz_ob[:,2], color = "orange")
ax.scatter(rdvz_ob[0,0], rdvz_ob[0,1], rdvz_ob[0,2], marker='x', color = "orange")
ax.scatter(rdvz_ob[-1,0], rdvz_ob[-1,1], rdvz_ob[-1,2], marker='o', color = "orange")

plt.axis('scaled')

# # ==== print result ====

print(f"DeltaV requred: {dv_required} km/s\t Intercept Velocity: {dv_at_rdvz} km/s")
print(f"Travel time: {(rdvz_time_end-rdvz_time_start)/(60*60*24):.2f} days\t eccentricity of intercept: {Rdvz.e:.2f}\t Period of intercept: {Rdvz.period/(60*60*24):.2f} days")
plt.show()

