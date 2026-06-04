
import jkat
from Structures.holistic_mass_solver import MassInterpolator
from typing import Callable

import math as m
import numpy as np
from scipy.optimize import minimize_scalar

M = MassInterpolator()
interp = M.interp


def interpolator_wrapper(dv0:float,dv1:float,dv2:float)->float:
    '''wrapper to ensure it works, INPUT IS IN KM/S'''
    try:
        return interp(np.array([dv0*1000,dv2*1000,dv1*1000]))[0]
    except: return 99_000

def helio_optim(park:jkat.Orbit, ISO:jkat.Orbit, max_time:float, detect_t:float):
    '''find the optimal trajectory for the heliocentric Orberth manoeuvre'''
    # interpolator = MassInterpolator()

    # # find apoapsis after detection:
    apo = park.t(m.pi)
    while apo > detect_t: apo -= park.T
    while apo < detect_t: apo += park.T
    


    # # find periapsis after that:
    peri = park.tp # find periapsis after ISO tp
    while peri < apo: peri += park.T

    rp, vp = park.vectors(0) # parking orbit periapsis
    def F(t):
        ri,vi = ISO.t2vectors(t)
        vl1,vl2 = jkat.trajectories.lambert(rp, ri, t-peri,park.mu)
        dv2 = np.linalg.norm(vl2-vi)

        # construct rotation:
        z = vl1.dot(rp)/rp.dot(rp) * rp
        vproj = vl1 - z
        vt = np.linalg.norm(vp) * vproj/np.linalg.norm(vproj) # same magnitude as vp
        rotated = jkat.orbit_from_rv(rp,vt,park.mu)
        dv0 = park.vvec(m.pi) - rotated.vvec(m.pi)

        dv1 = vl1-vt

        assert np.linalg.norm(rotated.rvec(0) - rp) < 1 # same periapsis
        assert np.linalg.norm(rotated.rvec(m.pi) - park.rvec(m.pi)) < 1 # same apoapsis
        assert abs(vt.dot(rp)) < 1e-5 # v is strictly tangential


        radial_angle = m.pi/2 - m.acos(
            rp.dot(dv1) / (np.linalg.norm(dv1) * np.linalg.norm(rp))
        )
        radial_burn = dv1.dot(rp)/rp.dot(rp) * rp


        dv1 = np.linalg.norm(dv1)
        
        return {
        "ts": peri,
        "te": t,
        'dv0': np.linalg.norm(dv0),
        "dv1": dv1,
        "dv2": dv2,
        'r': np.linalg.norm(ri),
        'radial': radial_angle,
        'rad_burn': np.linalg.norm(radial_burn),
        'ob': rotated
    }
    def w(t):
        try:
            res = F(t)
            # return res['dv0']
            return interpolator_wrapper(res['dv0'],res['dv1'],res['dv2'])
        except (ValueError, ArithmeticError, AssertionError): return m.inf
    
    
    t_opt = minimize_scalar(w,bounds=(peri, max_time)).x # type:ignore
    res = w(t_opt)
    res = F(t_opt)
    res['mass'] = interpolator_wrapper(res['dv0'],res['dv1'],res['dv2']) # add mass
    return res


def rotate_to_match(ob:jkat.Orbit, target:jkat.Orbit)->tuple[float, np.ndarray, jkat.Orbit]:

    htgt = target.hvec
    hob = ob.hvec
    eob = ob.evec

    # project:
    z = eob*htgt.dot(eob)/eob.dot(eob)
    htgt = htgt - z

    # figure out angle 
    angle = m.acos(htgt.dot(hob)/(np.linalg.norm(htgt)*np.linalg.norm(hob)))

    #
    if np.cross(hob,htgt).dot(eob) > 0:
        angle *= -1
    
    return angle, *jkat.trajectories.orbit_rotation(ob,angle,f=m.pi)




    
    

