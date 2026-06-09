'''
Using small impulses to simulate low thrust trajectories
'''


import jkat
from jkat.trajectories import lambert
from jkat.utils import propagate_vectors
from functools import partial
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from Rendezvous_dV_requirements import recreate_ISO
from contingency_analysis import get_data_earth
from tqdm import tqdm


def lt_lambert(r0vec:np.ndarray, rtgtvec:np.ndarray,
               v0vec:np.ndarray, vtgtvec:np.ndarray,
               t:float, mu:float, 
               acceleration:float,
               initial_impulse:float,
               fraction:int=1,
               plot:bool=False)->tuple[float,float]:
    ''' 
    Similar to the jkat lambert solver, but splits up the initial burn to approximate
    low-thrust trajectories,
    should give an upper bound on the low thrust dV, and approach the correct
    value for increased values of divisions.
    (Note! this value is probably not the optimal low-thrust trajectory, but a valid one)

    acceleration is assumed to be constant, and an initial High thrust impulse
    can also be given. (this number will not be inclued in the final result.)
    result is the (augmented) injection dV and final dV (which is assumed impulsive)
    #TODO: apply to final leg as well, and include acceleration increases?
    for now only does short way calculations
    '''

    def prop_and_demonstrate(r,v,dt,mu)->tuple[np.ndarray,np.ndarray]:

        ob = jkat.orbit_from_rv(r,v,mu)
        r,v = ob.t2vectors(dt)
        if plot:
            jkat.plot(ob, t_bounds=(0,dt), t=dt, stilts=False)
        return r,v

    # turn acceleration into km/s^2 from m/s^2
    acceleration /= 1000

    # initial impulsive setup:
    v1,v2 = lambert(r0vec,rtgtvec,t,mu)
    dv = v1 - v0vec
    dv_applied = dv/norm(dv) * min(initial_impulse,norm(dv))
    residual = dv - dv_applied
    if norm(residual)<1e-5: return 0,float(norm(v2-vtgtvec))
    dt:float = norm(residual)/(acceleration*fraction)# type:ignore floating vs float
    if dt > t: raise ArithmeticError("too low acceleration")
    # propagate step:
    p, v = prop_and_demonstrate(r0vec,v0vec + dv_applied, dt,mu) 
    t -= dt
    dv_spent = 0
    # start iterating:
    for _ in range(fraction*2): # if we go over this something is seriously wrong
        
        # step:
        v1, v2 = lambert(p, rtgtvec, t, mu)
        dv = v1 - v
        dv_applied = dv/norm(dv) * min(dt*acceleration,norm(dv))
        dv_spent += float(norm(dv_applied))
        residual = dv - dv_applied
        if norm(residual)<1e-5: return dv_spent,float(norm(v2-vtgtvec)) # end reached

        #else apply res
        p,v = prop_and_demonstrate(p, v + dv_spent, dt, mu)
        t -= dt
        # repeat
    else: raise ArithmeticError("lt_lambert failed to converge")

def lt_single(r0vec:np.ndarray, rtgtvec:np.ndarray,
               v0vec:np.ndarray, vtgtvec:np.ndarray,
               t:float, mu:float, 
               acceleration:float,
               initial_impulse:float,
               plot:bool = True)->tuple[float,float]:
    
    # impulse:
    # turn acceleration into km/s^2 from m/s^2
    acceleration /= 1000
    def prop_and_demonstrate(r,v,dt,mu)->tuple[np.ndarray,np.ndarray]:

        ob = jkat.orbit_from_rv(r,v,mu)
        r,v = ob.t2vectors(dt)
        if plot:
            jkat.plot(ob, t_bounds=(0,dt), t=dt, stilts=False, color='red')
        return r,v

    # initial impulsive setup:
    v1,v2 = lambert(r0vec,rtgtvec,t,mu, prograde=True)
    dv = v1 - v0vec
    dv_applied = dv/norm(dv) * min(initial_impulse,norm(dv))
    residual = dv - dv_applied
    if norm(residual)<1e-5: return 0,float(norm(v2-vtgtvec))
    dt:float = norm(residual)/(acceleration)# type:ignore floating vs float
    if dt > t: raise ArithmeticError("too low acceleration")
    # propagate step:
    p, v = prop_and_demonstrate(r0vec,v0vec + dv_applied, dt,mu) 
    t -= dt
    
    # "low" thrust:
    v1,v2 = lambert(p,rtgtvec, t, mu, prograde=True)
    dv1 = norm(v1-v)
    dv2 = norm(v2 - vtgtvec)

    prop_and_demonstrate(p,v1,t,mu)
    return dv1,dv2 # type:ignore


def single_cost_analysis(row:pd.Series, acceleration:float, impulse:float)->dict:
    '''find relative and absolute penalty of low thrust using single lt analysis,
    impulse in km/s, acceleration in m/s (units are fun!)'''

    ISO,_,_ = recreate_ISO(row)
    ts,te = row['ts'], row['te']
    r0,v0 = jkat.Earth.t2vectors(ts)
    r1,v1 = ISO.t2vectors(te)
    ion_given = row['dvi'] - impulse
    if ion_given <= 0: return {'relative':np.nan, 'absolute':np.nan}



    dv0,dv1 = lt_single(
        r0,r1,v0,v1,(te-ts), ISO.mu,acceleration, impulse, False
    )
    diff = (dv0 - ion_given + dv1 - row['dvr'])
    rel = diff/ion_given
    return {
        'absolute': diff,
        'relative': rel
    }


if __name__ == "__main__":



    df = get_data_earth()
    
    r = df['r'].to_numpy()/jkat.AU
    plt.hist(r)
    plt.show()
    
    
    input()
    
    df = df[df['dvi'] < 14.73]
    df = df[df['dvr'] < 5]
    print(f"{len(df)=}")
    rel = []
    abso = []
    failed = 0
    a = 9000 / (jkat.YEAR)
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            res = single_cost_analysis(row, a, 14.73 - 5.754/2) # impulse from run of Vesta
            relv = res['relative']
            absv = res['absolute']
            if relv > 10:
                print(f"outlier detected: {relv=}, {absv=}")
                continue

            rel.append(relv)
            abso.append(absv)
        except ArithmeticError: failed +=1

    print(f'{failed=}')
    print(f"rel-avg: {np.average(rel)}, rel-std: {np.std(rel)}")

    plt.hist(rel)
    plt.title("relative")
    plt.show()
    print(f"abs-avg: {np.average(abso)}, abs-std: {np.std(abso)}")
    plt.hist(abso)
    plt.title("absolute")
    plt.show()
    



    
