'''
Pytest tests for the orbit module, 
based on Curtis' exercises
'''
from .orbit import *
from .utilities import *
import pytest
import numpy as np
import math as m

def within_1_percent(a,b)->bool:
    # fpe check:
    if np.linalg.norm(a)<1e-8: return (np.linalg.norm(b) < 1e-8) # type:ignore
    return np.linalg.norm(a-b)/np.linalg.norm(a) < 0.01 #type:ignore

# ==== time and anomaly ===========

def test_kepler_universal_time_elliptical():
    h = EQ_RAD_EARTH*8  
    mu = SGP_EARTH
    for e in [0.0,0.3,0.5,0.7,0.9]:
        period = 2*m.pi*m.sqrt((h*h/(mu*(1-e*e)))**3/mu)
        tt = np.linspace(0,period,endpoint=False)
        for t in tt:
            M  = mu**2/(h**3) * (abs(e**2-1))**(3/2) * t
            kep = mean_2_true(M,e)
            uni = time_2_true(t,e,h,mu)
            assert within_1_percent(uni,kep)

def test_kepler_universal_time_hyper():
    h = EQ_RAD_EARTH*8  
    mu = SGP_EARTH
    period = 90*60
    for e in [1.0,1.5,2.0,3.0,4.0]:
        tt = np.linspace(-period,period,endpoint=False)
        for t in tt:
            M  = mu**2/(h**3) * (abs(e**2-1))**(3/2) * t if e != 1 else (mu**2/(h**3))*t 
            try:
                kep = mean_2_true(M,e)
            except: continue # can't compare is kepler doesn't work
            uni = time_2_true(t,e,h,mu)
            assert within_1_percent(uni,kep)

def test_kepler_universal_theta_elliptical():
    h = EQ_RAD_EARTH*8
    mu = SGP_EARTH
    for e in [0.0,0.3,0.5,0.7,0.9]:
        tt = np.linspace(0.01,2*m.pi,endpoint=False)
        for t in tt:
            uni = true_2_time(t,e,h,mu)
            try:
                M = true_2_mean(t,e)
            except: continue # can't compare is kepler doesn't work
            kep = M /( mu**2/(h**3) * (abs(e**2-1))**(3/2))
            assert within_1_percent(uni,kep)

def test_kepler_universal_theta_hyper():
    h = EQ_RAD_EARTH*8
    mu = SGP_EARTH
    for e in [1.0,1.5,2.0,3.0,4.0]:
        asymp_ang = m.acos(-1/e)
        tt = np.linspace(-asymp_ang + 0.05,asymp_ang - 0.05,endpoint=False)
        for t in tt:
            uni = true_2_time(t,e,h,mu)
            M = true_2_mean(t,e)
            kep = M /( mu**2/(h**3) * (abs(e**2-1))**(3/2)) if e != 1 else (
                M / (mu**2/(h**3)))
            assert within_1_percent(uni,kep)


def test_uni_round_trip_elliptical():
    h = EQ_RAD_EARTH*8
    mu = SGP_EARTH
    for e in [0.0,0.3,0.5,0.7,0.9]:
        tt = np.linspace(0,2*m.pi, 49,endpoint=False)
        for t in tt:
            time = true_2_time(t,e,h,mu)
            t2 = time_2_true(time,e,h,mu)
            time2 = true_2_time(t2,e,h,mu)
            assert within_1_percent(time,time2)
            assert within_1_percent(t,t2)

def test_uni_round_trip_hyper():
    h = EQ_RAD_EARTH*8
    mu = SGP_EARTH
    for e in [1.0,1.5,2.0,3.0,4.0]:
        asymp_ang = m.acos(-1/e)
        tt = np.linspace(-asymp_ang + 0.05,asymp_ang - 0.05,endpoint=False)
        for t in tt:
            time = true_2_time(t,e,h,mu)
            t2 = time_2_true(time,e,h,mu)
            time2 = true_2_time(t2,e,h,mu)
            assert within_1_percent(time,time2)
            assert within_1_percent(t,t2)

# ========= curtis tests ==================
class Test_Curtis_exercises:

    @pytest.fixture
    def fixt_ob(self)->Orbit:
        '''standard earth orbit'''
        return orbit_from_keplerian(EQ_RAD_EARTH,0,0,0,0,0,SGP_EARTH)


    def test_curtis_2_5(self,fixt_ob:Orbit):
        ob = fixt_ob
        ob.period = DAY
        assert within_1_percent(ob.apoapsis, EQ_RAD_EARTH + 35_786)
        assert within_1_percent(ob.tangential_v(0),3.075)
    
    def test_curtis_2_7(self, fixt_ob:Orbit):
        ob = fixt_ob
        ob.change_apses(4000+6378,400+6378)
        assert within_1_percent(ob.e, 0.2098)
        assert within_1_percent(ob.h, 57_172)
        assert within_1_percent(ob.a, 8578)
        assert within_1_percent(ob.period,2.196*(60*60))

    def test_curtis_2_9(self, fixt_ob:Orbit):
        ob = fixt_ob
        ob.e = 1
        ob.periapsis = 7000
        theta1 = ob.crosses_altitude(8000)
        assert not theta1 is None
        theta2 = ob.crosses_altitude(16000)
        assert not theta2 is None
        p1 = ob.theta_to_rv(theta1)[0]
        p2 = ob.theta_to_rv(theta2)[0]
        d = np.linalg.norm(p1-p2)
        assert within_1_percent(d,13_270)


    def test_curtis_2_10(self):
        
        r = np.array([14_600,0,0])
        v = elaz_vector(0,m.radians(90-50), 8.6)
        ob = orbit_from_rv(r,v,SGP_EARTH)
        assert within_1_percent(ob.h, 80_708)
        assert within_1_percent(ob.e,1.3393)
        assert within_1_percent(ob.periapsis, 6986)
        assert within_1_percent(ob.a, -20_590)
        assert within_1_percent(ob.C3,19.36)
        assert within_1_percent(ob.aiming_radius,18_340)



    def test_curtis_2_11(self, fixt_ob:Orbit):
        ob = fixt_ob
        ob.e = 0.3
        ob.h = 60_000
        r,v = ob.theta_to_rv(np.radians(120))
        assert within_1_percent(r, np.array([-5312.7, 9201.9, 0]))
        assert within_1_percent(v, np.array([-5.7533, -1.3287, 0]))

    def test_curtis_2_12(self):
        r = np.array([7000,9000,0])
        v = np.array([-3.3472,9.1251,0])
        ob = orbit_from_rv(r,v,SGP_EARTH)
        assert within_1_percent(ob.h,94_000)
        assert within_1_percent(ob.time_to_theta(0),m.radians(52.125))
        assert within_1_percent(ob.e,1.538)
    
    def test_curtis_2_13_and_14(self):
        r = np.array([8182.4,-6865.9,0])
        v = np.array([0.47572,8.8116,0])
        ob = orbit_from_rv(r,v,SGP_EARTH)
        t = ob.time_to_theta(0)
        t2 = t + m.radians(120)
        r,v = ob.theta_to_rv(t2)
        r_facit = np.array([1454.9, 8251.6,0])
        v_facit = np.array([-8.1323,5.6785, 0])
        assert within_1_percent(r,r_facit)
        assert within_1_percent(v,v_facit)
        assert within_1_percent(ob.e,1.0563)
        assert within_1_percent(t % (2*m.pi),m.radians(288.44))


    def test_curtis_3_1(self, fixt_ob:Orbit):
        ob = fixt_ob
        ob.change_apses(9600,21000)
        T = ob.theta_to_time(np.radians(120))
        assert within_1_percent(T, 4077)

    def test_curtis_3_2(self, fixt_ob:Orbit):
        ob = fixt_ob
        ob.change_apses(9600,21000)
        assert within_1_percent(ob.e, 0.37255)
        theta = ob.time_to_theta(3*60*60)
        assert within_1_percent(theta, np.radians(193.2)) # curtis answer wrong? (should be 195.8?)

    def test_curtis_3_4(self,fixt_ob:Orbit):
        ob = fixt_ob
        ob.e = 1
        # not testing the energy eq:
        ob.periapsis = 7972
        ob.t_p = 0
        th = ob.time_to_theta(6*60*60)
        r = ob.polar_equation(th)
        assert within_1_percent(r, 86_899)

    def test_curtis_3_5(self):
        ob = orbit_from_rv(np.array([6378+300,0,0]), np.array([0,15,0]),SGP_EARTH)
        T = ob.theta_to_time(np.radians(100))
        r = ob.polar_equation(np.radians(100))
        assert within_1_percent(T, 4141.4)
        assert within_1_percent(r,48_497)
        r,v = ob.time_to_rv(T + 3*60*60)
        assert within_1_percent(np.linalg.norm(r), 163_180)
        assert within_1_percent(np.linalg.norm(v), 10.51)

    def test_curtis_3_6(self,fixt_ob:Orbit):
        ob = fixt_ob
        ob.e = 1.4682
        ob.h = 95_154
        ob.link_time_and_theta(m.radians(30),0)
        assert within_1_percent(ob.polar_equation(m.radians(30)), 10_000)
        assert within_1_percent(ob.velocity(m.radians(30)), 10)
        t = ob.time_to_theta(1*60*60)
        assert within_1_percent(t % (2*m.pi),m.radians(100.04))

    def test_curtis_3_7(self):
        ob = orbit_from_rv(np.array([7000,-12_124, 0]), np.array([2.6679, 4.6210, 0]), SGP_EARTH)
        r,v = ob.time_to_rv(60*60)
        assert within_1_percent(r, np.array([-3296.8,7413.9,0]))
        assert within_1_percent(v, np.array([-8.2977,-0.96309,0]))


    def test_curtis_4_3(self):
        r = np.array([-6045, -3490, 2500])
        v = np.array([-3.457, 6.618, 2.533])
        ob = orbit_from_rv(r,v,SGP_EARTH)
        assert within_1_percent(ob.h, 58_310)
        assert within_1_percent(ob.i, np.radians(153.2))
        assert within_1_percent(ob.RAAN, np.radians(255.3))
        assert within_1_percent(ob.e, 0.1712)
        assert within_1_percent(ob.arg_p, np.radians(20.07))
        assert within_1_percent(ob.time_to_theta(0), np.radians(28.45))

    def test_curtis_4_7(self):
        
        ob = orbit_from_keplerian(-80_000,
                                  1.4,
                                  m.radians(30),
                                  m.radians(40),
                                  m.radians(60),
                                  m.radians(30),
                                  SGP_EARTH)
        ob.h = 80_000
        ob.link_time_and_theta(m.radians(30),0)
        r,v = ob.time_to_rv(0)
        r_facit = np.array([-4040,4815,3629])
        v_facit = np.array([-10.39, -4.772, 1.744])
        assert within_1_percent(r,r_facit)
        assert within_1_percent(v,v_facit)

    def test_curtis_5_2(self):
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
        assert within_1_percent(ob.time_to_theta(0), np.radians(350.8)) # time to theta is wrong

        
    def test_curtis_5_3(self):
        r1 = np.array([273_378,0,0])
        r2 = np.array([145_820,12_758,0])
        dt = 48_600
        ob = orbit_from_lambert(r1,r2,0,dt,SGP_EARTH)
        p_alt = ob.polar_equation(0)
        t_p2 = ob.t_p - dt
        assert within_1_percent(p_alt, 160.2+6378)
        assert within_1_percent(t_p2, 38_396)

    def test_curtis_6_1(self):
        # using trajectory optimizer
        origin = orbit_from_keplerian(1,0,0,0,0,0,SGP_EARTH)
        origin.change_apses(800+EQ_RAD_EARTH, 480+EQ_RAD_EARTH)
        destination = orbit_from_keplerian(1,0,0,0,0,0,SGP_EARTH)
        destination.a = (16_000+EQ_RAD_EARTH)*2
        destination.e = 0

        # TODO