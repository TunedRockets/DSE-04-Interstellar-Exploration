from .orbit import Orbit, orbit_from_ephemeris, plot_orbit
import matplotlib.pyplot as plt
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

Venus = orbit_from_ephemeris(
    0.72333566*AU,
    0.00677672,
    m.radians(3.39467605),
    m.radians(181.97909950),
    m.radians(131.60246718),
    m.radians(76.67984255),
    SGP_SUN
)

Mercury = orbit_from_ephemeris(
    0.38709843*AU,
    0.20563661,
    m.radians(7.00559432),
    m.radians(252.25166724),
    m.radians(77.45771895),
    m.radians(48.33961819),
    SGP_SUN
)

Saturn = orbit_from_ephemeris(
    9.53667594*AU,
    0.05386179,
    m.radians(2.48599187),
    m.radians(49.95424423),
    m.radians(92.59887831),
    m.radians(113.66242448),
    SGP_SUN
)

Uranus = orbit_from_ephemeris(
    19.18916464*AU,
    0.04725744,
    m.radians(0.77263783),
    m.radians(313.23810451),
    m.radians(170.95427630),
    m.radians(74.01692503),
    SGP_SUN
)

Neptune = orbit_from_ephemeris(
    30.06992276*AU,
    0.00859048,
    m.radians(1.77004347),
    m.radians(-55.12002969),
    m.radians(44.96476227),
    m.radians(131.78422574),
    SGP_SUN
)

Pluto = orbit_from_ephemeris(
    39.48211675*AU,
    0.24882730,
    m.radians(17.14001206),
    m.radians(238.92903833),
    m.radians(224.06891629),
    m.radians(110.30393684),
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

def get_solar_system_ax():
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(0,0,0, lw=3, color="red")
    # plot_orbit(ax, Mercury)
    # plot_orbit(ax, Venus)
    plot_orbit(ax, Earth)
    # plot_orbit(ax, Mars)
    plot_orbit(ax, Jupiter)
    # plot_orbit(ax, Saturn)
    # plot_orbit(ax, Uranus)
    # plot_orbit(ax, Neptune)
    # plot_orbit(ax, Pluto)
    return ax