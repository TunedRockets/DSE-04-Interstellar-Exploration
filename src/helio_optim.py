
import jkat
from typing import Callable

import math as m
import numpy as np



def helio_optim(park:jkat.Orbit, ISO:jkat.Orbit, max_time:float, boost_max:float):
    '''find the optimal trajectory for the heliocentric Orberth manoeuvre'''

    peri = park.tp # find periapsis after ISO tp
    while peri < ISO.tp: peri += park.T

    rp, vp = park.vectors(0) # parking orbit periapsis
    def F(t):
        ri,vi = ISO.t2vectors(t)
        vl1,vl2 = jkat.trajectories.lambert(rp, ri, t-peri,park.mu)
        dv2 = np.linalg.norm(vl2-vi)

        # construct rotation:
        z = vl1.dot(rp)/rp.dot(rp) * rp
        v = vl1 - z
        v = v*np.linalg.norm(vp)/np.linalg.norm(v) # same magnitude as vp
        rotated = jkat.orbit_from_rv(rp,v,park.mu)
        dv0 = park.vvec(m.pi) - rotated.vvec(m.pi)

        dv1 = np.linalg.norm(vl1-v)
        return {
        "ts": peri,
        "te": t,
        'dv0': np.linalg.norm(dv0),
        "dv1": dv1,
        "dv2": dv2,
        'r': np.linalg.norm(ri),
        'ob': rotated
    }
    def w(t):
        try:
            res = F(t)
            return res['dv2'] + res['dv0'] + (0 if res['dv1'] < boost_max else res['dv1']*1000) # heavily discourage dv1
        except (ValueError, ArithmeticError): return m.inf
    t_opt = minimizer_1d(w,peri, max_time)
    
    res = F(t_opt)
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



def minimizer_1d(f:Callable, a:float, b:float)->float:

    #preselect:
    max_step = (b-a)/20
    pp = np.arange(a,b,max_step)
    FF = []
    for p in pp: FF.append(f(p))
    pp = pp[np.argsort(FF)]

    epsilon = 1e-6
    alpha = 0.7

    for _ in range(1000):
        pF = f(p)
        dp = (pF - f(p-epsilon))/epsilon
        # pseudo newton but held back
        step = - pF/dp * alpha
        if abs(step) > max_step: step = max_step * np.sign(step)
        max_step = abs(step)

        p = p + step
        if step < epsilon: return p
    else: return p
    
    

