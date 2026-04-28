from .orbit import Orbit, orbit_from_ephemeris
from .utilities import SGP_SUN, AU, SGP_EARTH
import math as m
import datetime

# some standard objects (In the ICRF/J2000):
# only keplerian elements, does not take into account perturbations

Earth = orbit_from_ephemeris(
    1.00000261*AU,
    0.01671123,
    m.radians(-0.00001531),
    m.radians(100.46457166),
    m.radians(102.93768193),
    0,
    SGP_SUN
)
Mars = orbit_from_ephemeris(
    1.52371034*AU,
    0.09339410,
    m.radians(1.84969142),
    m.radians(-4.55343205),
    m.radians(-23.94362959),
    m.radians(49.55953891),
    SGP_SUN
)
Jupiter = orbit_from_ephemeris(
    5.20288700*AU,
    0.04838624,
    m.radians(1.30439695),
    m.radians(34.39644051),
    m.radians(14.72847983),
    m.radians(100.47390909),
    SGP_SUN
)

# Known ISOs (not including non-gravitational acceleration)
pe_to_p = lambda pe, e: pe*(1+e)
to_epoch = lambda year,month,day: (datetime.date(year,month,day) - datetime.date(2000,1,1)).total_seconds()

Omuamua = Orbit(
    p = pe_to_p(0.255916*AU, 1.20113),
    e = 1.20113,
    i = m.radians(122.74),
    arg_p = m.radians(241.811),
    RAAN = m.radians(24.597),
    t_p = to_epoch(2017,9,9), # 9th september 2017
    sgp=SGP_SUN
)

Borisov = Orbit(
    p = pe_to_p(2.00652*AU, 3.3565),
    e=3.3565,
    i=m.radians(44.053),
    RAAN=m.radians(308.15),
    arg_p=m.radians(209.12),
    t_p=to_epoch(2019,12,8),
    sgp=SGP_SUN
)

ATLAS = Orbit(
    p = pe_to_p(1.35645*AU, 6.14135),
    e=6.14135,
    i = m.radians(175.12),
    RAAN=m.radians(322.17),
    arg_p = m.radians(128.02),
    t_p=to_epoch(2025,10,29),
    sgp=SGP_SUN
)
