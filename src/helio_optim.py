
import jkat
from typing import Callable

import math as m
import numpy as np

def helio_optim(park:jkat.Orbit, ISO:jkat.Orbit, detect_t:float, bounds:tuple[float,float,float,float], boost_max:float):
    '''two stage heliocentric optimizer'''

    bounds = (max(detect_t, bounds[0]), bounds[1], bounds[2], bounds[3])

    peri = park.tp
    while peri < ISO.tp: peri += park.T

    def F(r):
        dv0, park2 = jkat.trajectories.orbit_rotation(park,r,f=m.pi)
        try: # periapsis burn
            retp = jkat.direct_transfer(park2,ISO,
                ts_min = peri - 100, ts_max = peri + 100, te_min = ISO.tp, te_max = ISO.tp + bounds[3], 
                dv1_w = 1, dv2_w = 1)
            wp = retp['dv2'] + np.linalg.norm(dv0) + retp['dv1']
        except (ArithmeticError, ValueError): wp = m.inf
        try: # non-peri burn
            reta = jkat.direct_transfer(park2,ISO,
                ts_min = peri - park2.T/2, ts_max = peri + park2.T/2, te_min = ISO.tp, te_max = ISO.tp + bounds[3], 
                dv1_w = 1, dv2_w = 1)
            wa = reta['dv2'] + np.linalg.norm(dv0) + reta['dv1']
        except (ArithmeticError, ValueError): wa = m.inf

        return min(wp,wa)
    
    points = np.linspace(-m.pi, m.pi, 10)
    Fpoints = []
    for p in points: Fpoints.append(F(p))
    points = points[np.argsort(Fpoints)]
    points = points[np.isfinite(Fpoints)]


    r_opt = golden_section_minimizer(F,-m.pi, m.pi, tol=m.radians(1))
    dv0, park2 = jkat.trajectories.orbit_rotation(park,r_opt,f=m.pi)
    retp = jkat.direct_transfer(park2,ISO,
                ts_min = peri - 100, ts_max = peri + 100, te_min = ISO.tp, te_max = ISO.tp + bounds[3], 
                dv1_w = 1, dv2_w = 1)
    reta = jkat.direct_transfer(park2,ISO,
                ts_min = peri - park2.T/2, ts_max = peri + park2.T/2, te_min = ISO.tp, te_max = ISO.tp + bounds[3], 
                dv1_w = 1, dv2_w = 1)
    if retp['dv2'] + np.linalg.norm(dv0) + retp['dv1'] < reta['dv2'] + np.linalg.norm(dv0) + reta['dv1']:
        ret = retp 
        ret['h_type'] = 'peri'
    else:
        ret = reta
        ret['h_type'] = 'api'

    ret.update({
        'dv0':dv0, "rot":r_opt
    })
    return ret

    


def golden_section_minimizer(f:Callable, a:float, b:float, tol:float=1e-2)->float:

    invphi =  (m.sqrt(5) - 1) / 2 

    while b-a > tol:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2
    
    

