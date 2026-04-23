'''
Interface with the Synthetic-population-of-Interstellar-Objects
package by Dusan Marceta.
'''

from lib.Synthetic_population_of_Interstellar_Objects.synthetic_population import synthetic_population
from .orbit import Orbit
from .utilities import SGP_SUN, AU
import numpy as np

def get_ISO(T:float=0)->list[Orbit]:
    '''Generates synthetic orbits of ISOs,
    If T is 0 (default), a snapshot of the population is generated,
    If T is a number (years), an expectation over that time
    is generated.
    rm is the sphere inside which the orbits generated'''
    #WIP
    # WHAT TIME TO USE???

    # CONSTANTS (sourced from example, case 1):
    rm = 10 # raduis of model sphere [AU]
    n0 = 0.1 # number density in interstellar space [AU^-1]
    v_min = 1e3 # max interstellar speed [m/s]
    v_max = 2e5 # min interstellar speed [m/s]
    u_sun = 1e4 # velocity components of sun
    v_sun = 1.1e4 # w.r.t. LSR [m/s]
    w_sun = 7e3
    sigma_vx = 3.1e4 # std-dev of ISO velocity
    sigma_vy = 2.3e4 # w.r.t. LSR [m/s]
    sigma_vz = 1.6e4
    vd = np.deg2rad(7) # vertex deviation [rad]
    va = 0 # asymmetric drift [m/s]
    R_reff = 696_340_000 # reference radius of sun [m]

    # q (periapsis) is in AU, rest is radians
    q, e, theta, inc, RAAN, arg_p = synthetic_population(T,
    rm, n0, v_min, v_max, u_sun, v_sun, w_sun, sigma_vx, sigma_vy, sigma_vz, va, vd, R_reff)

    # translate q to p:
    p = q*(1+e) * AU
    oobb = []
    for i in range(len(q)):
        ob = Orbit(p[i],e[i],inc[i],RAAN[i],arg_p[i],0,SGP_SUN)
        ob.link_time_and_theta(theta[i],0) # deal with time for longer somehow
        oobb.append(ob)
    return oobb



