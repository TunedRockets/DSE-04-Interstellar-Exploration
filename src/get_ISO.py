'''
Interface with the Synthetic-population-of-Interstellar-Objects
package by Dusan Marceta, and turn this into orbit objects and their detection times
for further analysis, interface is the get_ISO function, rest is supporting function for that
'''

from lib.Synthetic_population_of_Interstellar_Objects.synthetic_population import synthetic_population
from jkat import Orbit
from jkat.utils import SUN_MU, AU, YEAR, root_finder_bisection, pe2p
from jkat import Earth
import numpy as np
import math as m
from tqdm import tqdm
from typing import Callable
from functools import partial
from multiprocessing import Pool

LSST_sensitivity_magnitude = 24.38



def job(obtuple, gen_type, lsst)->tuple[Orbit,float,str]|None:
    q,e,i,raan,argp = obtuple
    p = pe2p(q*AU,e)
    ob = Orbit(p,e,i,raan,argp,0,SUN_MU)
    # shuffle times:
    ob.tp = np.random.rand()*YEAR
    # figure out detection:
    try:
        if gen_type == 'sun': # debug, always let through
            d_time = -m.inf
        else:
            H,gen_type = _generate_abs_magnitude(gen_type=gen_type)
            d_time = _detection_time(ob, H, lsst)
    except (ArithmeticError, ValueError):
        # wasn't detected. skip
        return;

    return (ob, d_time, gen_type)



def get_ISO(T:float=0, rm:float=10, gen_type:str='')->list[tuple[Orbit, float,str]]:
    '''Use Marčeta's model for ISO generation to create a batch of synthetic ISOs. 

    :param T: Time passed to the synthetic_population model. 
    If 0, will return a snapshot at one point in time. for our analysis, keep as 0.\n defaults to 0
    :type T: float, optional
    :param rm: size in AU of the generating sphere in the synthetic_population model.
    generated objects will necessarily have their periapsis inside the sphere.
    for our analysis, keep as 10.\n defaults to 10
    :type rm: float, optional
    :param gen_type: what type of generation function is used for the absolute magnitude,
    options are 'omuamua' or 'atlas-borisov', if omitted will randomize for each ISO.\n defaults to ''
    :type gen_type: str, optional
    :return: list of tuples containing the ISO orbit, time of detection (in same epoch as ISO orbit), and type of generation function
    :rtype: list[tuple[Orbit, float,str]]
    '''
    
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
    F = partial(job, gen_type=gen_type, lsst=LSST_sensitivity_magnitude)

    # q (periapsis) is in AU, rest is radians
    q, e, theta, inc, RAAN, arg_p = synthetic_population(T,
    rm, n0, v_min, v_max, u_sun, v_sun, w_sun, sigma_vx, sigma_vy, sigma_vz, va, vd, R_reff)

    obtuples = zip(q,e,inc,RAAN,arg_p)
    
    with Pool() as p:
        res = filter(None,tqdm(p.imap_unordered(F, obtuples), desc="Detecting ISOs from Marčeta", total=len(q)))
        oobb = list(res)
    
    
    print(f"\t{len(oobb)}/{len(q)} orbits were detected and passed on to analysis")
    return oobb

generation_types = ['omuamua', 'atlas-borisov', 'sun']
def _generate_abs_magnitude(gen_type:str='')->tuple[Callable[[float],float],str]:
    '''generate a absolute magnitude function for use in figuring out detection distance.
    takes in distance (in km) and returns absolute magnitude

    :param gen_type:what type of generation function is used for the absolute magnitude,
    options are 'omuamua' or 'atlas-borisov', if omitted will randomize for each ISO.\n
    :type gen_type: str, optional
    :return: Absolute magnitude function, takes in orbital height and returns absolute magnitude
    :rtype: tuple[Callable[[float],float],str]
    '''
    # ref:
    # - 'Omuamua: ~22.4
    # - Borisiv: ~12 (including coma)
    # - ATLAS ~12.5  (including coma)
    gen_type = gen_type.lower()
    if not gen_type in generation_types:
        gen_type = generation_types[np.random.randint(len(generation_types)-1)]
    
    match gen_type:

        case 'omuamua':
            H:Callable[[float],float] = lambda r: 22.4 # no brightening with distance 
        case 'atlas-borisov':
            H:Callable[[float],float] = lambda r: 12.5 # no brightening with distance
            # technically it should brighten by 1-2 magnitudes as it moves into the system, but this doesn't impact
            # the detection times that much so is ignored
        case 'sun':
            H:Callable[[float],float] = lambda r: -2 # very bright
    return H, gen_type

def _HG_magnitude(ob:Orbit, time:float, absolute_magnitude:float)->float:
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

    r_e = Earth.t2rvec(time)
    r_ob = ob.t2rvec(time)
    r_delta = r_ob - r_e
    au_delta = np.linalg.norm(r_delta)/AU
    au_ob = np.linalg.norm(r_ob)/AU

    phi = m.acos(r_delta.dot(-r_e)/(np.linalg.norm(r_e)*np.linalg.norm(r_delta)))
    varphi1 = m.exp(-A1 * m.tan(phi/2)**B1)
    varphi2 = m.exp(-A2 * m.tan(phi/2)**B2)
    phase = 2.5*m.log10((1-G)*varphi1 + G*varphi2)

    V = absolute_magnitude + 5*m.log10(au_delta) + 5*m.log10(au_ob) - phase
    return V

def _detection_time(ob:Orbit, absolute_magnitude:Callable[[float],float], sensitivity:float)->float:
    '''figure out time of detection for the given ISO orbit, an absolute magnitude function, and the sensitivity of the detecting telescope(s)

    :param ob: Orbit of the ISO
    :type ob: Orbit
    :param absolute_magnitude: Absolute magnitude function
    :type absolute_magnitude: Callable[[float],float]
    :param sensitivity: sensitivity of the telescope(s)/survey
    :type sensitivity: float
    :return: time of detection to an accuracy of 1 second
    :rtype: float
    '''

    # excess magnitude (negative means detected)
    F = lambda t: _HG_magnitude(ob,t,absolute_magnitude(ob.r(ob.f(t)))) - sensitivity

    enter_system = ob.cross_radius(5*AU)
    if m.isnan(enter_system): raise ArithmeticError("does not enter inner system")
    e_time = ob.t(-enter_system)
    p_time = ob.f(0)
    
    # already detected?
    if F(e_time) < 0:
        e2_time = e_time - YEAR # look further back
        while F(e2_time) < 0: e2_time -= YEAR
        return root_finder_bisection(F,e2_time, e_time, tolerance=1) # look in outer system
    # else find detection time:
    F_low = F(p_time)
    if not (cross_earth:=ob.cross_radius(AU)) is None: # check when it crosses earth
        x1_time = ob.t(-cross_earth)
        x2_time = ob.t(cross_earth)
        x1F = F(x1_time)
        x2F = F(x2_time)
        if F_low > 0 and x1F < 0:
            p_time = x1_time
            F_low = x1F
        elif F_low > 0 and x2F < 0:
            p_time = x2_time
            F_low = x2F
            
    if F_low > 0: # same sign, bad
        raise ArithmeticError(f"min magnitude is {F_low}, which is still positive (should be negative), periapsis is: {ob.periapsis/AU} AU")
    d_time = root_finder_bisection(F,e_time,p_time,tolerance=1) # within 1 second
    return d_time


