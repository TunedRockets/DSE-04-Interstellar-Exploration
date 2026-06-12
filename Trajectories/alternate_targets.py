''' 
What other things can we reach???
'''
import jkat
import datetime as dt
import numpy as np
from jkat.ephemeris.JPLHorizons import horizons_request
from scipy.optimize import minimize



ts_mis = jkat.ephemeris.to_time(dt.datetime(2036,1,1))
te_mis = jkat.ephemeris.to_time(dt.datetime(2046,1,1))
travel_time = jkat.YEAR * 10


def can_reach(osc_ob:jkat.Orbit, origin:jkat.Orbit)->bool|np.ndarray:
    '''can we reach the given orbit? (search space is the 10 years
    of our mission, 2036-2046), with a 10 year travel time'''
    

    def residual(dvi,dvr):
        dvi = max(0,(dvi - 12))
        dvr += dvi*2
        return 10 - dvr

    def ion_req(t): # try to minimize this
        ts = t[0]; te = t[1]
        try:
            r1,v1 = origin.t2vectors(ts)
            r2,v2 = osc_ob.t2vectors(ts)

            try: vl1,vl2 = jkat.trajectories.lambert(r1,r2,te-ts,origin.mu, True)
            except: vl1=vl2=np.array([np.inf,np.inf,np.inf])
            try: va1, va2 = jkat.trajectories.lambert(r1,r2, te-ts, origin.mu, False)
            except: va1=va2=np.array([np.inf,np.inf,np.inf])
            dvl1 = np.linalg.norm(v1-vl1)
            dvl1 = max(dvl1,0)
            dvl2 = np.linalg.norm(v2-vl2)

            dva1 = np.linalg.norm(v1-va1)
            dva1 = max(dva1,0)
            dva2 = np.linalg.norm(v2-va2)
            l = residual(dvl1, dvl2)
            a = residual(dva1,dva2)

            return -max(l,a)
        except: return np.inf 


    x0 = prescan_opt(
        ion_req,
        np.linspace(ts_mis, te_mis,10),
        np.linspace(ts_mis + travel_time, te_mis + travel_time, 10)
    )

    topt = minimize(ion_req, x0, bounds=(
        (ts_mis, te_mis), (ts_mis + travel_time, te_mis + travel_time)
    ))

    if topt.success and ion_req(topt.x) > 0: return topt.x
    else: return False

    

def prescan_opt(F, xx, yy):
    '''prescan for a sorta global minima of the function'''

    xg, yg = np.meshgrid(xx,yy)
    xg = xg.flatten(); yg = yg.flatten();
    ww = []
    for i in range(len(xg)):
        ww.append(F((xg[i],yg[i])))
    idx = np.array(ww).argmin()
    return (xg[idx],yg[idx])


def test_and_show(name:str):
    tgt = horizons_request(
        name
    )
    tgt = tgt.osculating_orbit(jkat.ephemeris.to_time(dt.datetime(2036,1,1)))
    origin = jkat.Earth
    if (t := can_reach(tgt, origin)) is False: 
        print("can't reach :(")
        exit()
    # else:
    print("Success!")
    jkat.add_solar_system(t[0], '11111111', True) # type:ignore
    trans = jkat.orbit_from_lambert_transfer(origin, tgt, t[0], t[1], True) # type: ignore
    jkat.plot(trans, t_bounds=(t[0],t[1])) # type: ignore
    jkat.plot(tgt, t=t[1], t_bounds=(ts_mis, te_mis + travel_time*2)) # type: ignore
    jkat.show()

def beat_other_DSE():
    planet9 = jkat.Orbit(550*jkat.AU, 0.2, 0.4, 3.3, 2.7, 0, jkat.SUN_MU)
    origin = jkat.Earth
    if (t := can_reach(planet9, origin)) is None: 
        print("can't reach Planet 9 :(")
        exit()
    # else:
    print("Planet 9 Success!")
    jkat.add_solar_system(t[0], '11111111', True) # type:ignore
    trans = jkat.orbit_from_lambert_transfer(origin, planet9, t[0], t[1], True) # type: ignore
    jkat.plot(trans, t_bounds=(t[0],t[1])) # type: ignore
    jkat.plot(planet9, t=t[1], t_bounds=(ts_mis, te_mis + travel_time*2)) # type: ignore
    jkat.show()

def catch_up(i:int):
    ISO = [jkat.examples.Omuamua, jkat.examples.Borisov, jkat.examples.ATLAS][i-1]
    origin = jkat.Earth
    if (t := can_reach(ISO, origin)) is None: 
        print("can't reach :(")
        exit()
    # else:
    print(f"Success with {i}I!")
    jkat.add_solar_system(t[0], '11111111', True) # type:ignore
    try: trans = jkat.orbit_from_lambert_transfer(origin, ISO, t[0], t[1], True) # type: ignore
    except: trans = jkat.orbit_from_lambert_transfer(origin, ISO, t[0], t[1], False) # type: ignore
    print(jkat.ephemeris.from_time(t[1]))
    print(np.linalg.norm(ISO.t2rvec(t[1]))/jkat.AU)
    jkat.plot(trans, t_bounds=(t[0],t[1])) # type: ignore
    jkat.plot(ISO, t=t[1], t_bounds=(ts_mis, te_mis + travel_time*2)) # type: ignore
    jkat.show()


if __name__ == "__main__":

    targets = [
       'Pluto',
       'Eris',
       'Haumea',
       'Quaoar',
       'Makemake',
       'Gonggong',
       'Sedna',
       'Orcus',
       'Ixion',
       'Varda',
       'Ceres',
       'vesta',
       'eros',
       'pallas',
       'interamnia',
       '90000001', # Halley
       'hale-bopp'
    ]
    # beat_other_DSE()
    # catch_up(1)
    # catch_up(2)
    # catch_up(3)
    test_and_show('hale-bopp')

    for n in targets:
        tgt = horizons_request(n)
        tgt = tgt.osculating_orbit(jkat.ephemeris.to_time(dt.datetime(2036,1,1)))
        origin = jkat.Earth
        r = can_reach(tgt,origin)
        if r is None: print(f"{n} couldn't be reached")
        # else:
        print(f"{n} is possible")
       