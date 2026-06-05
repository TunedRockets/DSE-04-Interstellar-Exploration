
import jkat
from Structures.holistic_mass_solver import MassInterpolator
from typing import Callable

import math as m
import numpy as np
from scipy.optimize import minimize_scalar, minimize

M = MassInterpolator()
interp = M.interp


def interpolator_wrapper(dv0:float,dv1:float,dv2:float)->float:
    '''wrapper to ensure it works, INPUT IS IN KM/S'''
    try:
        return interp(np.array([dv0*1000,dv2*1000,dv1*1000]))[0]
    except: return (dv1*5 + dv2 + dv0*7)*10_000 + 99_000



def mad_optim(ISO:jkat.Orbit, max_time:float, detect_t:float, vinf:float):

    def F(t):
        try:
            '''manually for own weighting'''
            t1 = t[0]; t2 = t[1]
            r1, v1 = jkat.Earth.t2vectors(t1)
            r2, v2 = ISO.t2vectors(t2)
            try: vl1,vl2 = jkat.trajectories.lambert(r1,r2,t2-t1,ISO.mu, True)
            except: vl1=vl2=np.array([np.inf,np.inf,np.inf])
            try: va1, va2 = jkat.trajectories.lambert(r1,r2, t2-t1, ISO.mu, False)
            except: va1=va2=np.array([np.inf,np.inf,np.inf])
            dvl1 = np.linalg.norm(v1-vl1) - vinf
            dvl1 = max(dvl1,0)
            dvl2 = np.linalg.norm(v2-vl2)
            lmass = interpolator_wrapper(0,dvl1,dvl2) #type: ignore

            dva1 = np.linalg.norm(v1-va1) - vinf
            dva1 = max(dva1,0)
            dva2 = np.linalg.norm(v2-va2)
            amass = interpolator_wrapper(0,dva1,dva2)#type: ignore
            if lmass < amass:
                return {
                "ts": t1,
                "te": t2,
                'dv0': 0,
                "dv1": dvl1,
                "dv2": dvl2,
                'r': np.linalg.norm(r2),
                'mass': lmass
            }
            else: return {
                "ts": t1,
                "te": t2,
                'dv0': 0,
                "dv1": dva1,
                "dv2": dva2,
                'r': np.linalg.norm(r2),
                'mass': amass
            }
        except: return {'dv0':m.inf, 'dv1': m.inf, 'dv2': m.inf, 'mass': m.inf}
    
    def w(t): 
        try: return F(t)['mass']
        except(ValueError, ArithmeticError): return m.inf

    x0 = np.array(((ISO.tp + ISO.tp + jkat.YEAR)/2, (ISO.tp + jkat.YEAR + ISO.tp + 2*jkat.YEAR)/2))
    topt = minimize(w, x0, bounds=((detect_t,max_time), (detect_t, max_time)))
    if topt.success: return F(topt.x)
    else: return {}
    

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
            return interpolator_wrapper(res['dv0'],res['dv1'],res['dv2'])
        except (ValueError, ArithmeticError, AssertionError): return m.inf


    t_opt = minimize_scalar(w,bounds=(peri, max_time)).x # type:ignore
    res = w(t_opt)
    res = F(t_opt)
    res['mass'] = interpolator_wrapper(res['dv0'],res['dv1'],res['dv2']) # add mass
    return res

def check_if_possible(dv0_budget:float, dv1_budget:float, dv2_budget:float, park:jkat.Orbit, ISO:jkat.Orbit, max_time:float, detect_t:float):
    '''Finds if the ISO is reachable within the available delta V budgets (km/s)'''

    # find apoapsis after detection:
    apo = park.t(m.pi)
    while apo < detect_t: apo += park.T

    # find periapsis after that:
    peri = park.tp  # find periapsis after ISO tp
    while peri < apo: peri += park.T

    rp, vp = park.vectors(0)  # parking orbit periapsis

    def F(t):
        ri, vi = ISO.t2vectors(t)
        vl1, vl2 = jkat.trajectories.lambert(rp, ri, t - peri, park.mu)
        dv2 = np.linalg.norm(vl2 - vi)

        # construct rotation:

        z = vl1.dot(rp) / rp.dot(rp) * rp
        vproj = vl1 - z
        vt = np.linalg.norm(vp) * vproj / np.linalg.norm(vproj)  # same magnitude as vp
        rotated = jkat.orbit_from_rv(rp, vt, park.mu)
        dv0 = park.vvec(m.pi) - rotated.vvec(m.pi)

        dv1 = vl1 - vt

        assert np.linalg.norm(rotated.rvec(0) - rp) < 1  # same periapsis
        assert np.linalg.norm(rotated.rvec(m.pi) - park.rvec(m.pi)) < 1  # same apoapsis
        assert abs(vt.dot(rp)) < 1e-5  # v is strictly tangential

        radial_angle = m.pi / 2 - m.acos(
            rp.dot(dv1) / (np.linalg.norm(dv1) * np.linalg.norm(rp))
        )
        radial_burn = dv1.dot(rp) / rp.dot(rp) * rp

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
            dv0 = res["dv0"]
            dv1 = res["dv1"]
            dv2 = res["dv2"]
            # print(dv0)
            return max(0, (dv0-dv0_budget)) + max(0, (dv1-dv1_budget)) + max(0, (dv2-dv2_budget))

        except:
            return m.inf

    t_solution = minimizer_1d(w,peri, max_time)
    res = F(t_solution)
    # print((w(t_solution)==0))
    # print(w(t_solution))
    # print(res)
    return (w(t_solution)==0), res






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



def minimizer_1d(f:Callable, a:float, b:float, tol:float = 1e-4, escape_value:float|None = None)->float:

    invphi = (m.sqrt(5) - 1) / 2  #

    # golden section search:
    while b - a > tol:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        fc = f(c); fd = f(d)
        if escape_value is not None:
            if fc<escape_value: break
        if fc < fd or fd==np.inf:
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2


