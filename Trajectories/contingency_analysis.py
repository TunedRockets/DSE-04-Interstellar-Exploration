'''
Using a direct from earth trajectory, with a comparable 
Vinf impluse as would be required for the heliocentric case.
Uses a dV penalty for the Ion stage to simulate low-thrust trajectory
(since the majority of the burn, up to escape velocity speed, is impulsive,
this should be broadly accurate).
here, dv0 is the boost stage dv, dv1 is the extra required by the ions,
and dv2 is the rendezvous burn.

using the analysis suited for the 


TODO: need better heat shield sizer!!! (check with Sem)


'''
import jkat
import pandas as pd
import numpy as np
import math as m
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import minimize
from Rendezvous_dV_requirements import find_best_point, get_ISO, interpolator_wrapper, _under, Hestia
from pathlib import Path
from Structures.holistic_mass_solver import Vesta
from scipy.interpolate import RegularGridInterpolator

VINF = 8.5 # + 0.577 
MAX_MISSION_TIME = 15 # [years]
MAX_BOOST_DV = 5 # [km/s]
LOW_THRUST_PENALTY = 2 # extra cost for low thrust (made up number)
AREA_OF_INTEREST = (MAX_BOOST_DV,20,20) # Area where the mass function is defined


# create simple mass interpolator.
INTERP:RegularGridInterpolator = None # type:ignore

def _job(p):
    dv1 = p[0]; dv2 = p[1]
    try:
        V = Vesta(0,dv2*1000,dv1*1000)
        V._converge()
        return V.lower_stage_wet_mass
    except (ArithmeticError,ValueError) as e: 
        return m.nan


def make_interp(res:int = 20):
    '''interpolator strictly for Vesta model'''
    xx = np.linspace(0,AREA_OF_INTEREST[0], res) # boost
    yy = np.linspace(0,(AREA_OF_INTEREST[1] + AREA_OF_INTEREST[2]) * 3/4, res) # Ion

    xg,yg = np.meshgrid(xx,yy)
    xg.flatten(); yg.flatten()
    pp = np.column_stack((xg,yg))
    with mp.Pool() as p:
        mm = tqdm(p.map(_job, pp, 4),desc="creatign interpolator")


    mm = np.reshape(np.array(mm),(res,res),)
    global INTERP
    INTERP = RegularGridInterpolator((xx,yy), mm)
    return;

def earth_mass(dv0,dv1,dv2):
    '''translate dv0:boost, dv1+dv2:ion into the interpolator'''
    # if INTERP is None:
    #     make_interp(); return earth_mass(dv0,dv1,dv2)
    # else:
    #     try: 
    #         return INTERP(dv0,dv1+dv2)
    #     except: return (dv0*5+dv1+dv2)*10_000 + 999_000

    # Vesta does not want to run the power analysis :(

    return interpolator_wrapper(0,dv0,dv1+dv2)

def study_ISO(ISO:jkat.Orbit, detect_t:float)->dict:
    '''study earth based intercept'''
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
            dvl1 = np.linalg.norm(v1-vl1) - VINF
            dvl1 = max(dvl1,0)
            dvl2 = np.linalg.norm(v2-vl2)

            dva1 = np.linalg.norm(v1-va1) - VINF
            dva1 = max(dva1,0)
            dva2 = np.linalg.norm(v2-va2)

            # Balance out burns, here dv0 is boost, dv1 is low_thrust as part of initial
            # and dv2 is rendezvous
            def massfn(v1,v2):
                '''give maximum of MAX_BOOST to v1, rest to v2, which matches using impulsive + low thrust'''
                if v1 > MAX_BOOST_DV:
                    v2 += (v1 - MAX_BOOST_DV)*LOW_THRUST_PENALTY
                    v1 = MAX_BOOST_DV
                return earth_mass(v1,0,v2) # don't need to split up before sending it to function
            
            lmass = massfn(dvl1, dvl2)
            amass = massfn(dva1,dva2)
            if lmass < amass:
                return {
                "ts": t1,
                "te": t2,
                'dv0': min(dvl1, MAX_BOOST_DV),
                "dv1": max((dvl1 - MAX_BOOST_DV)*LOW_THRUST_PENALTY, 0),
                "dv2": dvl2,
                'r': np.linalg.norm(r2),
                'mass': lmass
            }
            else: return {
                "ts": t1,
                "te": t2,
                'dv0': min(dva1, MAX_BOOST_DV),
                "dv1": max((dva1 - MAX_BOOST_DV)*LOW_THRUST_PENALTY, 0),
                "dv2": dva2,
                'r': np.linalg.norm(r2),
                'mass': amass
            }
        except: return {'dv0':m.inf, 'dv1': m.inf, 'dv2': m.inf, 'mass': m.inf}
    
    def w(t): 
        try: return F(t)['mass']
        except(ValueError, ArithmeticError): return m.inf

    x0 = np.array(((ISO.tp + ISO.tp + jkat.YEAR)/2, (ISO.tp + jkat.YEAR + ISO.tp + 2*jkat.YEAR)/2))
    topt = minimize(w, x0, bounds=((detect_t,detect_t + MAX_MISSION_TIME*jkat.YEAR), (detect_t, detect_t + MAX_MISSION_TIME*jkat.YEAR)))
    if topt.success: return F(topt.x)
    else: return {}


def job(ISOtuple:tuple[jkat.Orbit, float, str])->dict:

    np.seterr(all="ignore")
    ISO, detect_t, g_type = ISOtuple

    detect_r = ISO.r(ISO.f(detect_t))/jkat.AU
    out = {"detection_r":detect_r, "periapsis":ISO.periapsis/jkat.AU, "magnitude_generation_method": g_type,
        'time_until_periapsis':(ISO.tp - detect_t)/jkat.DAY,
            "parameter":ISO.p, "e":ISO.e, "i":ISO.i, "RAAN":ISO.raan, "arg_p":ISO.argp, "t_p":ISO.tp, 
            "ISO_excess_velocity":ISO.vinf}
    try:
        out.update(study_ISO(ISO, detect_t))
    except (ArithmeticError, ValueError, AssertionError): return out
    return out

def study_batch_earth(gen_type:str='', N_batches:int=20)->pd.DataFrame:
    '''multithreaded analysis'''
    
    ISOs = get_ISO(gen_type=gen_type, N_batches=N_batches)
    #for each ISO get row:
    resl = []
    with mp.Pool() as p:
    
        res = tqdm(p.imap_unordered(job, ISOs), desc=f"Studying ISOs, (from earth)", total=len(ISOs))
        resl = list(res)
    return pd.DataFrame(resl)




def direct_earth_analysis(N_batches:int=30):
    '''same as before but now direct from earth'''
    N = 350
    Paim = 0.9 # probability aim

    df = study_batch_earth(N_batches=N_batches)

    print("interesting fraction:")
    dfi = df[df["dv0"] <= AREA_OF_INTEREST[0]]
    dfi = dfi[dfi["dv1"] <= AREA_OF_INTEREST[1]]
    dfi = dfi[dfi["dv2"] <= AREA_OF_INTEREST[2]]
    print(dfi[['dv0','dv1','dv2','mass']])
    count = len(dfi)

    print(f" {count} / {len(df)} = {count/len(df)}")

    point = find_best_point(df, N, Paim, AREA_OF_INTEREST, earth_mass)
    if point is None: s = 'no valid points'
    else:
        v0 = point[0]
        v1 = point[1]
        v2 = point[2]
        m = point[3]
        changed = False
        # get accurate mass:
        if v0 > AREA_OF_INTEREST[0] or v1 > AREA_OF_INTEREST[1] or v2 > AREA_OF_INTEREST[2]:
            try:
                H = Vesta(v0*1000, v2*1000, v1*1000, False,0.001, )
                H._converge()
                m = H.lower_stage_wet_mass
                changed = True
            except: m = np.inf


        under = _under(df, v0,v1,v2)
        P = under/len(df)
        P = (1-(1-P)**N)

        s = 'E:'

        s += (f"best mass: {m:>7.0f}" + ('*' if changed else " ") + "kg," +
            f"success chance: {P:>4.2%}, " +
            f"delta vees: ({VINF:3.1f}+) {v0:>6.3f}, {v1:>6.3f} (/{LOW_THRUST_PENALTY:3.1f}), {v2:>6.3f} km/s, " +
            f"ISOs generated: {count:>3}/{len(df):<5}"
        )
    path = Path(__file__).parent / 'runs.txt'
    with open(path, 'a') as file:
        file.write(s + '\n')
        print(s)
    return




if __name__ == '__main__':

    make_interp()
    print(earth_mass(5,7,7))


    # direct_earth_analysis(500)