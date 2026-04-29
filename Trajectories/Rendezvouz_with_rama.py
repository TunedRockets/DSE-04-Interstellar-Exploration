'''
Proof of concept code for calculating ISO interceptor trajectories
using the Kepler orbit script
'''
from src.orbit import Orbit, trajectory_optimizer, plot_orbit
from src.utilities import *
from src.get_ISO import get_ISO
from src.examples import Earth
import matplotlib.pyplot as plt
import numpy as np


detect_distance = 5*AU
max_time = 10*YEAR
weight = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}


ISOs = get_ISO(rm=2)

for ISO in ISOs:
    detect_theta = ISO.crosses_altitude(detect_distance)
    if detect_theta is None: 
        print("Not inside detect distance")
        continue
    detect_time = ISO.theta_to_time(-detect_theta)

    try:
        insert_dv, rdvz_dv,st,et,er = trajectory_optimizer(Earth,ISO,detect_time,detect_time+max_time, **weight)
    except:
        print("Omtimizser did not converge")
        continue
    print(f'{insert_dv=}\n{rdvz_dv=}\n{st=}\n{et=}\ntravel time={(et-st)/DAY} days\ner={er/AU} AU')

    ax = plt.figure().add_subplot(projection='3d')
    
    intercept = Orbit.orbit_from_lambert(Earth.time_to_rv(st)[0],
                                         ISO.time_to_rv(et)[0],
                                         st,
                                         et,
                                         Earth.sgp)
    plot_orbit(ax, intercept, et,(intercept.time_to_theta(et)-intercept.time_to_theta(st)),color="red")

    plot_orbit(ax,Earth, color="blue")
    plot_orbit(ax,ISO, max_alt=max(ISO.polar_equation(ISO.time_to_theta(et)),er,AU), time=et, color="green")
    plt.axis('scaled')
    plt.show()