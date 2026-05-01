'''
Pytest tests for the orbit module, 
based on Curtis' exercises
'''
from .orbit import Orbit, orbit_from_rv, orbit_from_lambert
from .utilities import SGP_EARTH
import numpy as np

def within_1_percent(a,b)->bool:
    return np.linalg.norm(a-b)/np.linalg.norm(a) < 0.01 #type:ignore

def test_curtis_2_7():
    ob = Orbit(1,0,0,0,0,0,SGP_EARTH)
    ob.change_apses(4000+6378,400+6378)
    assert within_1_percent(ob.e, 0.2098)
    assert within_1_percent(ob.h, 57_172)
    assert within_1_percent(ob.a, 8578)
    assert within_1_percent(ob.period,2.196*(60*60))

def test_curtis_2_11():
    ob = Orbit(1,0.3,0,0,0,0,SGP_EARTH)
    ob.h = 60_000
    r,v = ob.theta_to_rv(np.radians(120))
    assert within_1_percent(r, np.array([-5312.7, 9201.9, 0]))
    assert within_1_percent(v, np.array([-5.7533, -1.3287, 0]))


def test_curtis_3_1():
    ob = Orbit(1,0,0,0,0,0,SGP_EARTH)
    ob.change_apses(9600,21000)
    T = ob.theta_to_time(np.radians(120))
    assert within_1_percent(T, 4077)

def test_curtis_3_2():
    ob = Orbit(1,0,0,0,0,0,SGP_EARTH)
    ob.change_apses(9600,21000)
    assert within_1_percent(ob.e, 0.37255)
    theta = ob.time_to_theta(3*60*60) % 2*np.pi
    assert within_1_percent(theta, np.radians(193.2)) # curtis answer wrong? (should be 195.8?)

def test_curtis_3_5():
    ob = orbit_from_rv(np.array([6378+300,0,0]), np.array([0,15,0]),SGP_EARTH)
    T = ob.theta_to_time(np.radians(100))
    r = ob.polar_equation(np.radians(100))
    assert within_1_percent(T, 4141.4)
    assert within_1_percent(r,48_497)
    r,v = ob.time_to_rv(T + 3*60*60)
    assert within_1_percent(np.linalg.norm(r), 163_180)
    assert within_1_percent(np.linalg.norm(v), 10.51)

def test_curtis_3_7():
    ob = orbit_from_rv(np.array([7000,-12_124, 0]), np.array([2.6679, 4.6210, 0]), SGP_EARTH)
    r,v = ob.time_to_rv(60*60)
    assert within_1_percent(r, np.array([-3296.8,7413.9,0]))
    assert within_1_percent(v, np.array([-8.2977,-0.96309,0]))


def test_curtis_4_3():
    r = np.array([-6045, -3490, 2500])
    v = np.array([-3.457, 6.618, 2.533])
    ob = orbit_from_rv(r,v,SGP_EARTH)
    assert within_1_percent(ob.h, 58_310)
    assert within_1_percent(ob.i, np.radians(153.2))
    assert within_1_percent(ob.RAAN, np.radians(255.3))
    assert within_1_percent(ob.e, 0.1712)
    assert within_1_percent(ob.arg_p, np.radians(20.07))
    assert within_1_percent(ob.time_to_theta(0), np.radians(28.45))

def test_curtis_5_2():
    r1 = np.array([5000,10_000,2100])
    r2 = np.array([-14_600,2500,7000])
    dt = 60*60
    ob = orbit_from_lambert(r1,r2,0,dt,SGP_EARTH)
    assert within_1_percent(ob.h, 80_470)
    assert within_1_percent(ob.a, 20_000)
    assert within_1_percent(ob.e, 0.4335)
    assert within_1_percent(ob.RAAN, np.radians(44.60))
    assert within_1_percent(ob.arg_p, np.radians(30.71))
    assert within_1_percent(ob.i, np.radians(30.19))
    assert within_1_percent(ob.polar_equation(0), 4952+6378)
    assert within_1_percent(ob.t_p, 256.1)
    assert within_1_percent(ob.time_to_theta(0) % 2*np.pi, np.radians(350.8)) # time to theta is wrong

    
def test_curtis_5_3():
    r1 = np.array([273_378,0,0])
    r2 = np.array([145_820,12_758,0])
    dt = 48_600
    ob = orbit_from_lambert(r1,r2,0,dt,SGP_EARTH)
    p_alt = ob.polar_equation(0)
    t_p2 = ob.t_p - dt
    assert within_1_percent(p_alt, 160.2+6378)
    assert within_1_percent(t_p2, 38_396)