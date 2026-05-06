'''
Interface with the Synthetic-population-of-Interstellar-Objects
package by Dusan Marceta.
'''

from lib.Synthetic_population_of_Interstellar_Objects.synthetic_population import synthetic_population
from .orbit import Orbit
from .utilities import SGP_SUN, AU, YEAR, root_finder_bisection
from .examples import Earth
import numpy as np
import math as m
from tqdm import tqdm

LSST_sensitivity_magnitude = 24.38


def get_ISO(T:float=0, rm:float=10, gen_type:str='')->list[tuple[Orbit, float, float,str]]:
    '''Generates synthetic orbits of ISOs,
    If T is 0 (default), a snapshot of the population is generated,
    If T is a number (years), an expectation over that time
    is generated.
    rm is the sphere inside which the orbits generated'''

    # CONSTANTS (sourced from example, case 1):
    rm = rm # radius of model sphere [AU]
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
    for i in tqdm(range(len(q)), desc="Converting Marčeta ISOs to Keplerian orbits and determining detection time"):
        ob = Orbit(p[i],e[i],inc[i],RAAN[i],arg_p[i],0,SGP_SUN)

        # shuffle times:
        ob.t_p = np.random.rand()*YEAR
        # ob.link_time_and_theta(theta[i],0) # deal with time for longer somehow
        
        # figure out detection:
        try:
            H,gen_type = generate_abs_magnitude(gen_type=gen_type)
            d_time = detection_time(ob, H, LSST_sensitivity_magnitude)
        except (ArithmeticError, ValueError):
            # wasn't detected. skip
            continue


        oobb.append((ob, d_time, H, gen_type))
    print(f"\t{len(oobb)}/{len(p)} orbits were detected and passed on to analysis")
    return oobb


generation_types = ['omuamua', 'atlas-borisov', 'asteroidal', 'cometary']
def generate_abs_magnitude(gen_type:str='')->tuple[float,str]:
    '''(somehow) generate a random magnitude for the ISO
    return magnitude and generation type

    :return: absolute magnitude
    :rtype: float
    '''
    # ref:
    # - 'Omuamua: ~22.4
    # - Borisiv: ~12 (including coma)
    # - ATLAS ~12.5  (including coma)
    if gen_type == '':
        gen_type = generation_types[np.random.randint(len(generation_types))]
    

    match gen_type.lower():

        case 'omuamua':
            H = 22.4
        case 'atlas-borisov':
            H = 12.5
        case _:
            H = 15
            gen_type = 'fallback generation'



    return H, gen_type

def HG_magnitude(ob:Orbit, time:float, absolute_magnitude:float)->float:
    '''return the apparent magnitude of an orbit as seen from earth given an
    absolute magnitude

    :param ob: orbit in question
    :type ob: Orbit
    :param time: time in question
    :type time: float
    :param absolute_magnitude: absolute magnitude of the object
    :type absolute_magnitude: float
    :return: apparent magnitude
    :rtype: float
    '''


    # HG constants:
    A1 = 3.332
    A2 = 1.862
    B1 = 0.631
    B2 = 1.218
    G = 0.15

    r_e = Earth.time_to_rv(time)[0]
    r_ob = ob.time_to_rv(time)[0]
    r_delta = r_ob - r_e
    au_delta = np.linalg.norm(r_delta)/AU
    au_ob = np.linalg.norm(r_ob)/AU

    phi = m.acos(r_delta.dot(-r_e)/(np.linalg.norm(r_e)*np.linalg.norm(r_delta)))

    varphi1 = m.exp(-A1 * m.tan(phi/2)**B1)
    varphi2 = m.exp(-A2 * m.tan(phi/2)**B2)
    
    phase = 2.5*m.log10((1-G)*varphi1 + G*varphi2)

    V = absolute_magnitude + 5*m.log10(au_delta) + 5*m.log10(au_ob) - phase
    return V

def detection_time(ob:Orbit, absolute_magnitude:float, sensitivity:float)->float:

    # excess magnitude (negative means detected)
    F = lambda t: HG_magnitude(ob,t,absolute_magnitude) - sensitivity

    enter_system = ob.crosses_altitude(5*AU)
    if enter_system is None: raise ArithmeticError("does not enter inner system")
    e_time = ob.theta_to_time(-enter_system)
    p_time = ob.time_to_theta(0)

    # already detected?
    if F(e_time) < 0:
        e2_time = e_time - YEAR # look further back
        while F(e2_time) < 0: e2_time -= YEAR
        return root_finder_bisection(F,e2_time, e_time, tolerance=1) # look in outer system
    # else find detection time:

    # assume if it's visible it's visible at periapsis
    F_low = F(p_time)
    if F_low > 0: # same sign, bad
        raise ArithmeticError(f"min magnitude is {F_low}, which is still positive (should be negative), periapsis is: {ob.periapsis/AU} AU")
    d_time = root_finder_bisection(F,e_time,p_time,tolerance=1) # within 1 second
    return d_time


