import jkat


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))


from src.get_ISO import get_ISO
from src.orbit import Orbit
from src2.orbit import oberth_effect_optimzer
import math as m
import numpy as np

def translation_of_shame(ob:Orbit)->jkat.Orbit:

    return jkat.Orbit(
        ob.p,
        ob.e,
        ob.i,
        ob.RAAN,
        ob.arg_p,
        ob.t_p,
        ob.sgp
    )

def back_translation_of_shame(ob:jkat.Orbit)->Orbit:
    return Orbit(
        ob.p,ob.e,ob.i,ob.raan,ob.argp,ob.tp,ob.mu
    )

from jkat.utils import elements
a,e = elements.apse2ae(5.45*jkat.AU, 10*695_700)

park = jkat.orbit_from_ephemeris(
    a, e, m.radians(1.3), 0, m.radians(124.14), m.radians(100.4), jkat.SUN_MU
)
ISO = jkat.Orbit(
69225117.196624,
1.013101,
0.112718,
3.142535,
5.792523,
27973063.401473,
    jkat.SUN_MU
)

_, _, transfer_orbit, st, et, er = oberth_effect_optimzer(
                back_translation_of_shame(ISO), # type:ignore
                park.rvec(0),
                np.linalg.norm(park.vvec(0)), # type:ignore
                park.tp,
                ISO.tp - 5*jkat.JULIAN_YEAR,
                ISO.tp + 10*jkat.JULIAN_YEAR,
                optimize_rendezvous=True,
                # tp_window_width=10*YEAR/365
            ) 

jkat.plot(translation_of_shame(transfer_orbit), t_bounds=(st,et), max_distance=(er + jkat.AU), color="purple", t=et) # type:ignore
jkat.plot(ISO, max_distance=(er + jkat.AU), color="deeppink")  # type:ignore
jkat.plot(park, color="purple")
# obs = get_ISO(gen_type='atlas-borisov')
# for obb in obs:
#     ob,td,_ = obb
#     ob = translation_of_shame(ob)
#     jkat.plot(ob, t_bounds=(-m.inf, 0), stilts=False, max_distance=8*jkat.AU, t=-10, pw=1, color='gray')

jkat.add_solar_system()
jkat.show()
