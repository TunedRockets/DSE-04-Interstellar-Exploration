'''
main file for handling the orbit part.
Algorithms primarily sourced from:
- Orbital Mechanics for Engineering Students by Howard D. Curtis
- Fundamentals of Astrodynamics and Applications by David Vallado

By: Johannes Nilsson
Created: 2025-06-11
'''
import numpy as np
from .utilities import *
from typing import Callable
import matplotlib.pyplot as plt
import math as m
from tqdm import tqdm
import warnings



# ===========================

class Orbit():
    '''
    Main class to represent a keplerian orbit
    values of the orbit are stored as keplerian elements,
    p,e,i,RAAN,arg_p, and t_p.\n
    as such, a parent body must be represented in the form of a given sgp.
    other variables such as a (semi-major-axis), h (specific angular momentum), the apsides and the period
    can be accessed and modified.
    special cases, I.e. circular, planar, parabolic, and degenerate, 
    are dealt with specially:
    - Degenerate orbits are currently no implemented
    - Circular orbits assume the "periapsis" is the ascending node
    - planar orbits assume the "ascending node" is on the X axis
    - Circular planar orbits apply both of the above
    - orbits with negative inclination are possible, and are interpreted with the "ascending node"
      being the descending one
    
    '''

    def __init__(self, p:float, e:float, i:float, RAAN:float,
                  arg_p:float, t_p:float, sgp:float ) -> None:
        '''
        Generator for the Orbit class. Directly specify the 6 state variables
        used internally for the orbit, and the parent sgp

        :param p: Parameter (semi-latus rectum)
        :type p: float
        :param e: Eccentricity
        :type e: float
        :param i: Inclination
        :type i: float
        :param RAAN: Right ascension of the ascending node
        :type RAAN: float
        :param arg_p: Argument of periapsis
        :type arg_p: float
        :param t_p: Time of periapsis passage
        :type t_p: float
        :param sgp: Parent body standard gravitational parameter
        :type sgp: float
        '''


        self.p = p
        '''parameter (semi-latus rectum)'''
        self.e = e
        '''eccentricity'''
        self.i = i
        '''inclination'''
        self.RAAN = RAAN
        '''Right ascension of the ascending node'''
        self.arg_p = arg_p
        '''argument of periapsis'''
        self.t_p = t_p
        '''time of periapsis passage'''
        self.sgp = sgp
        '''Parent body standard gravitational parameter'''
    
    # === basic variations of the above (and changing them) ======

    @property
    def a(self)->float:
        '''Semi-major axis\n
        hyperbolic orbits are considered to have negative semi-major axes\n
        
        Changing this variable scales p by a corresponding amount\n
        only works if new a is same sign as old a'''
        if self.e != 1: return self.p/(1-self.e**2)
        else: return m.inf # parabolas have infinite semi major axes

    @a.setter
    def a(self, a:float)->None:
        if self.a * a > 0 and self.e != 1: 
            # don't change sign accidentally
            self.p = a*(1-self.e**2)
        else: raise ValueError("Cannot flip sign of semi-major axis")
        
    @property
    def h(self)->float:
        '''angular momentum\n
        changing this changes the parameter of the orbit'''
        return np.sqrt(self.p*self.sgp)
    
    @h.setter
    def h(self, h:float):
        # p = h^2/mu
        self.p = m.sqrt(h)/self.sgp

    @property
    def periapsis(self)->float:
        return self.p/(1+self.e)
    
    @periapsis.setter
    def periapsis(self, pe:float)->None:
        '''The lowest point of the orbit.\n
        Changing this changes p and e to set new periapsis, keeping apoapsis for elliptic orbits
        and e for para/hyperbolic orbits'''
        if self.apoapsis == m.inf: # para/hyper:
            # pe = p/(1+e) => p = pe(1+e)
            self.p = pe*(1+self.e)
        else:
            self.change_apses(new_pe=pe)

    @property
    def apoapsis(self)->float:
        if self.e < 1: return self.p/(1-self.e)
        else: return m.inf # since apoapsis is infinitely far away

    @apoapsis.setter
    def apoapsis(self, ap:float)->None:
        '''The highest point of the orbit. returns inf for para/hyperbolic orbits\n
        Changing this changes p and e to set new apoapsis, keeping periapsis.
        raises ValueError on change if para/hyperbolic'''
        if self.e >= 1: raise ValueError("Cannot change apoapsis of Hyperbolic orbit (since it doesn't exist)")
        self.change_apses(new_ap=ap)

    @property
    def period(self)->float:
        '''Period of orbit, Changing this changes p by way of a, unless it is hyperbolic in which 
        case it raises a ValueError'''
        if self.e < 1: return 2*m.pi*m.sqrt(self.a**3/self.sgp)
        else: return m.inf # since there is no period
    
    @period.setter
    def period(self, period):
        if self.e >= 1: raise ValueError("Cannot change period of Hyperbolic orbit (since it doesn't exist)")
        self.a = m.cbrt(self.sgp * (period/(2*m.pi))**2)
        # (T/2pi)^2 = a^3/mu 

    @property
    def mean_motion(self)->float:
        '''mean motion of the orbit, in (mean anomaly) radians / second
        '''
        return m.sqrt(self.sgp/(self.a**3))

    def change_apses(self, new_ap:float|None = None, 
                     new_pe:float|None = None)->float:
        ''' change apses of the orbit, keeping the apse line the same.
        returns the new eccentricity. necessarily less than one\n
        new values may be omitted to keep the current ones\n
        #TODO cover the case when ap < pe (swap arg_p around (means swapping t_p))
        to change it into a hyperbolic orbit use something else #TODO'''

        # insert already existing:
        if new_ap == None: new_ap = self.apoapsis
        if new_ap == m.inf: 
            raise ValueError(f"New apoapsis can't be infinite"
            "(did you forget to include apoapsis when changing hyperbolic orbit?)")

        if new_pe == None: new_pe = self.periapsis

        # figure out e:
        e = (new_ap - new_pe)/(new_ap + new_pe)

        # figure out parameter:
        p = 0.5*(new_ap + new_pe)*(1-e**2)

        self.e = e
        self.p = p

        # TODO: currently, this messes up the "periapsis at ascending node" scheme
        # if e = 0, maybe have a look at that

        return e 

    def polar_equation(self,theta)->float:
        '''simple polar equation that returns r for a given theta.
        (numpy vector compatible)'''
        return self.p/(1+self.e*np.cos(theta))

    def asymptote_angle(self)->float:
        '''returns the (external) angle to the asymptotes 
        of the hyperbolic orbit\n
        I.e. the angle between the asymptote and the periapsis'''
        if self.e <= 1: 
            raise ArithmeticError(f"Only Hyperbolic orbits have asymptotes,"
                                  f" {self.e=} is not asymptotic")
        return m.acos(-1/self.e)
    
    # ====== time ======

    def time_after_periapsis_to_theta(self, time)->float:
        '''calculate time after periapsis to what theta it should be
        (wraps a utilities function)'''
        
        return time_2_true(time, self.e, self.h, self.sgp)

    def time_to_theta(self,time:float)->float:
        '''calculate using epoch time what theta it should be
        (wraps a utilities function)'''
        return self.time_after_periapsis_to_theta(time - self.t_p)

    def theta_to_time_after_periapsis(self, theta:float)->float:
        '''calculate time after periapsis from given theta
        (wraps a utilities function)'''
        return true_2_time(theta, self.e, self.h, self.sgp)
    
    def theta_to_time(self,theta:float)->float:
        '''calculate epoch time from given theta
        (wraps a utilities function)'''
        return self.theta_to_time_after_periapsis(theta) + self.t_p
    
    def link_time_and_theta(self,theta,time)->None:
        '''Changes the time of periapsis passage so that the given theta happens at the given time'''

        t_theta = self.theta_to_time_after_periapsis(theta)
        self.t_p = time - t_theta
        return
    
    # === other orbits =====

    def synodic_period(self,other:"Orbit")->float:
        '''period between when the angle of the two orbiting objects is the same.
        I.e. the period between launch windows (on circular orbits)

        :param other: other orbit
        :type other: Orbit
        :return: synodic period
        :rtype: float
        '''
        if self.e >=1 or other.e >= 1:
            raise ValueError("hyperbolic orbits do not have synodic periods")
        return (self.period*other.period)/abs(self.period-other.period)
    
    def hohmann_angle(self,other:"Orbit")->float:
        '''calculates the optimal angle the other planets needs to be ahead
        for optimal hohmann transfer.

        :param other: other orbit
        :type other: Orbit
        :return: optimal angle
        :rtype: float
        '''
        if self.e >=1 or other.e >= 1:
            raise ValueError("hyperbolic orbits do not have hohmann transfers")
        # in optimal transfer, transfer orbit will cover pi radians,
        # during which time the orbit coveres pi - phi radians (phi=optimal)
        # thus n2*t = pi - phi
        # => phi = pi - n2*t
        # where n2 is mean motion of other, and t is transfer orbit time
        t = m.pi * m.sqrt((self.a + other.a)**3/(8*self.sgp))
        return m.pi - other.mean_motion*t
    
    def hohmann_time(self,other:"Orbit")->float:
        '''compute the optimal time for a hohmann transfer,
        assumes circular orbits so will be incorrect for ellitical orbits.
        this time repeats every synodic period

        :param other: other orbit
        :type other: Orbit
        :return: time of hohmann transfer
        :rtype: float
        '''
        if self.e >=1 or other.e >= 1:
            raise ValueError("hyperbolic orbits do not have hohmann transfers")
        
        # establish base angle:
        phi_0 = other.time_to_theta(0) - self.time_to_theta(0)
        phi_opt = self.hohmann_angle(other)
        effective_mean_motion = other.mean_motion - self.mean_motion

        # want to wait to when it goes from phi_0 to phi'
        # phi = phi_0 + n*t (mod 2pi)
        t_until = (phi_opt - phi_0)/effective_mean_motion
        return t_until
    
    # ================= getting vectors ====================
    @property
    def e_vec(self)->np.ndarray:
        '''eccentricity vector, points in the direction of periapsis'''
        return self.pqw_basis[:,0]*self.e # p vector scaled by e

    @property
    def h_vec(self)->np.ndarray:
        '''angular momentum vector, points out of the orbital plane'''
        return self.pqw_basis[:,2]*self.h # w vector scaled by h

    @property
    def pqw_basis(self)->np.ndarray:
        '''the perifocal frame basis coordinates as a column matrix [p,q,w] in the global coordinates\n
        also works as a transformation matrix from pqw to ijk (global),
        I.e. x_glob = Q*x_peri'''

        # in effect the combined rotation matrix of 
        # rot3(-RAAN)rot1(-i)rot3(-arg_p)
        # from Vallado
        
        # normalize angles:
        i = self.i
        RAAN = self.RAAN
        arg_p = self.arg_p
        if i < 0:
            i *= -1
            RAAN += m.pi
            arg_p += m.pi

        # for ease of writing:
        co = m.cos(self.RAAN)
        so = m.sin(self.RAAN)
        ci = m.cos(self.i)
        si = m.sin(self.i)
        cp = m.cos(self.arg_p)
        sp = m.sin(self.arg_p)

        Q = np.array([
            [co*cp - so*sp*ci, -co*sp -so*cp*ci, so*si],
            [so*cp+co*sp*ci, -so*sp + co*cp*ci, -co*si],
            [sp*si, cp*si, ci]
        ])
        return Q

    def time_to_rv(self,time:float)->tuple[np.ndarray, np.ndarray]:
        '''
        Returns the position and velocity vectors at a specified time
        '''
        theta = self.time_to_theta(time)
        return self.theta_to_rv(theta)

    def theta_to_rv(self,theta:float)->tuple[np.ndarray, np.ndarray]:
        '''
        returns the position and velocity vectors at a specified anomaly\n
        invalid angles (for hyperbolas) raise ValueErrors
        '''

        # first check for invalid angles:
        if self.e > 1:
            if self.asymptote_angle() < abs(theta):
                raise ArithmeticError(f"Invalid true anomaly, orbit is" 
                f"hyperbolic so has a max anomaly of {self.asymptote_angle()}")
            
        r = self.polar_equation(theta) * np.array([m.cos(theta), m.sin(theta), 0]) # r vector in pqw
        v = (self.sgp/self.h)*np.array([-m.sin(theta), self.e + m.cos(theta), 0]) # v vector in pqw
        r = self.pqw_basis@r
        v = self.pqw_basis@v

        return r, v

    # === misc ========

    def __repr__(self) -> str: # always need a repr (for debugging at least)
        return f"Orbit:\n {self.p=}\n {self.e=}\n {self.i=}\n {self.RAAN=}\n {self.arg_p=}\n {self.t_p=}\n {self.sgp=}"
    
    def normalize(self)->None:
        '''normalizes the orbit, making t_p shift to within one period, making inclination strictly positive,
        and ensuring all values are within their bounds (i.e. 0-2pi)'''

        # inclination fix:
        if self.i < 0:
            self.RAAN += m.pi
            self.arg_p += m.pi
            self.i *= -1
        # wrap orbital elements:
        self.RAAN %= 2*m.pi
        self.arg_p %= 2*m.pi
        self.i %= 2*m.pi
        if self.e < 1:
            self.t_p %= self.period
        return;

    def crosses_altitude(self, altitude:float)-> float|None:
        '''Checks if orbit crosses a certain altitude,
        and if so, at what anomalies\n
        returns:anomaly of collision (the positive one. other collision is the negative of this, None if no collision)'''

        if not (self.periapsis < altitude < self.apoapsis):
            return None # does not cross
        
        # reverse polar equation:
        return m.acos((self.p/altitude - 1)/self.e)

    def impact_point(self, radius:float)->np.ndarray|bool:
        """
        Returns the impact point coordinates in latitude, longitude of the current trajectory.\n
        Will return the impact site (when orbit going down), not the exit site (when orbit going up)
        """

        # check for impact exists:
        if not (anomaly := self.crosses_altitude(radius)):
            return False

        r = self.theta_to_rv(-anomaly)[0] # minus for impact

        coord = vector_elazr(r)[0:2] # grab elevation azimuth
        return coord #type:ignore

    def point_locus(self, theta1:float = -np.pi, theta2:float = np.pi, num_points:int = 360)->np.ndarray:
        '''Returns an array of points from and too a certain theta.\n
        useful for i.e. plotting the orbit\n
        (if orbit is hyperbolic it will softly limit the range)\n
        (this is better than using theta_to_rv since it doesn't generate v)\n
        impossible trajectories are given a value of inf'''

        # find hyperbolic ranges (if applicable)
        try:
            limit_theta = self.asymptote_angle() - 0.0001 # since the asymptote itself is infinity
        except ArithmeticError:
            limit_theta = m.inf
        
        while theta2 < theta1: theta2 += 2*m.pi # fix wraparound issues
        
        thetas = np.linspace(max(-limit_theta,theta1), min(limit_theta,theta2), num_points) # order the points

        points = []
        pqw = self.pqw_basis

        # TODO: replace with vectorized numpy method
        for theta in thetas:
            r = self.polar_equation(theta) * np.array([np.cos(theta), np.sin(theta), 0]) # r vector in pqw
            r = pqw@r
            points.append(r)

        return np.array(points)

    # == deprecated ==
    
    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def orbit_from_rv(r:np.ndarray, v:np.ndarray, sgp:float, time:float=0)->"Orbit":
        '''
        Creates an orbit given a position and velocity vector.\n
        the time is given to set the orbit within the epoch, if nothing is provided then
        it's assumed the given data is at the epoch.
        '''
        return orbit_from_rv(r,v,sgp,time)

    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def orbit_from_lambert(r1:np.ndarray, r2:np.ndarray, start_time:float,
                           end_time:float, sgp:float, short_way:bool = True)->"Orbit":
        '''creates an orbit by solving lambert's problem\n
        start_time is the time at r1, end_time is the time at r2. can choose between long and short way.\n
        for getting only the vectors, use lambert_vectors()\n
        start_time also used to set orbit in proper epoch'''
        return orbit_from_lambert(r1,r2,start_time,end_time,sgp,short_way)

    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def orbit_from_gauss(observations:list[np.ndarray],
                            times:list[float], 
                            positions:list[np.ndarray],
                            sgp:float)->"Orbit":
        '''Uses Gauss' method to find an orbit from 3 observations, taken at 3 different times.
        you also need to include the positions of the three observations at those 3 given times
        observations assumed to be normal vectors'''
        return orbit_from_gauss(observations,times,positions,sgp)
    
    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def from_ephemeris(a:float, e:float, i:float, L:float, long_p:float, RAAN:float, sgp:float)->"Orbit":
        '''creates an orbit from ephemeris numbers instead.
        a: semi-major axis
        e: eccentricity
        i: inclination
        L: mean longitude
        long_p: longitude of perihelion
        RAAN: longitude of ascending node.
        only works with elliptical orbits'''
        return orbit_from_ephemeris(a,e,i,L,long_p,RAAN,sgp)

    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def point_to_point(p1:np.ndarray, p2:np.ndarray, radius:float, start_time:float, end_time:float, sgp:float, angular_speed:float, epoch_angle:float)->"Orbit":
        '''Create a point to point orbit between two coordinates on a sphere with given radius\n
        points given in elevation/azimuth.\n
        essentially a wrapper of the lambert function'''
        return point_to_point(p1,p2,radius,start_time,end_time,sgp,angular_speed,epoch_angle)

    @staticmethod
    @warnings.deprecated("Static methods are moved to functions outside the class")
    def lambert_vectors(r1_vec:np.ndarray, r2_vec:np.ndarray, time:float, sgp:float, short_way:bool = True)->tuple[np.ndarray, np.ndarray]:
        '''Solves lamberts problem of finding an orbit from two position vectors and a time between them\n
        can choose between "short" or "long" way (defaults to short), does not consider solutions of multiple periods.\n
        this version returns only the initial and final velocity vectors,
        use lambert_orbit() to obtain the orbit itself (a combination of this and from_rv)'''
        return lambert_vectors(r1_vec, r2_vec, time, sgp, short_way)


# ==========================================================


# ======= Orbit Generators ===============
'''
This is functions that generate orbits from certain observations or data
previously static methods were included in the orbit class itself, but was moved here for clarity.
each function will return an orbit object.
methods for finding optimal transfers are not included, and are written below this section
'''

def orbit_from_ephemeris(a:float, e:float, i:float, L:float, long_p:float, RAAN:float, sgp:float)->Orbit:
    '''
    creates an orbit from ephemeris values, only works with elliptical orbits
    :param a: semi-major axis
    :type a: float
    :param e: eccentricity
    :type e: float
    :param i: inclination
    :type i: float
    :param L: mean longitude
    :type L: float
    :param long_p: longitude of perihelion
    :type long_p: float
    :param RAAN: longitude of ascending node
    :type RAAN: float
    :param sgp: Parent body standard gravitational parameter
    :type sgp: float
    '''
    if e >= 1: raise ValueError("Hyperbolic ephemeris not implemented")
    p = a*(1-e*e)
    M = L - long_p # mean anomaly
    arg_p = long_p - RAAN
    t_p = -M * m.sqrt(a**3/sgp)
    return Orbit(p,e,i,RAAN,arg_p,t_p,sgp)

def orbit_from_keplerian(a:float, e:float, i:float, RAAN:float, arg_p:float, theta:float, sgp:float)->Orbit:
    '''
    Generate an orbit from the 6 keplerian elements. true anomaly assumed to be
    the anomaly at the epoch, (from which the periapsis passage is calculated)

    :param a: Semi-major axis
    :type a: float
    :param e: Eccentricity
    :type e: float
    :param i: Inclination
    :type i: float
    :param RAAN: Right ascension of the ascending node
    :type RAAN: float
    :param arg_p: Argument of periapsis
    :type arg_p: float
    :param theta: True anomaly
    :type theta: float
    :param sgp: Parent body standard gravitational parameter
    :type sgp: float
    '''
    p = a*(1-e*e)
    ob = Orbit(p,e,i,RAAN,arg_p,0,sgp)
    ob.link_time_and_theta(theta,0) # fix true anomaly
    return ob

def orbit_from_rv(r:np.ndarray, v:np.ndarray, sgp:float, time:float=0)->Orbit:
    '''
    Creates an orbit given a position and velocity vector.\n
    the time is given to set the orbit within the epoch, if nothing is provided then
    it's assumed the given data is at the epoch.
    '''

    if np.linalg.norm(np.cross(r,v)) == 0:
        raise NotImplementedError("Degenerate orbits are not implemented yet")

    h_vec = np.cross(r,v) # angular momentum vector
    r_mag = np.linalg.norm(r)
    v_r = r.dot(v)/r_mag
    h = np.linalg.norm(h_vec) # angular momentum
    e_vec = np.cross(v,h_vec)/sgp - r/r_mag # eccentricity vector
    e = np.linalg.norm(e_vec) # eccentricty
    p = h_vec.dot(h_vec)/sgp # parameter
    k_hat = np.array([0,0,1]) # "up" vector
    n_vec = np.cross(k_hat,h_vec) # node vector

    # normal calculations:
    i = m.acos(h_vec[2]/h) # inclination

    RAAN = m.acos(n_vec[0]/np.linalg.norm(n_vec)) # RAAN
    if n_vec[1] < 0: RAAN = 2*m.pi - RAAN # angle is negative (but we want to keep it in range 0-2pi)
    
    arg_p = m.acos(n_vec.dot(e_vec)/(np.linalg.norm(n_vec)*e)) # argument of periapsis
    if e_vec[2] < 0: arg_p = 2*m.pi - arg_p

    theta = m.acos(e_vec.dot(r)/(e*r_mag)) # true anomaly
    if v_r < 0: theta = 2*m.pi - theta

    # special cases:
    if i==0 and e!= 0: # elliptical equitorial
        RAAN = 0 # "ascending node" on x axis
        arg_p = m.acos(e_vec[0]/e) # arg_p from x axis
        if e_vec[1] < 0: arg_p = 2*m.pi - arg_p

    elif e == 0 and i != 0: # circular inclined
        
        arg_p = 0 # "periapsis" on node
        theta = m.acos(n_vec.dot(r)/(r_mag*np.linalg.norm(n_vec)))
        if r[2] < 0: theta = 2*m.pi - theta
    elif e==0 and i== 0: # circular equatorial
        # node and periapsis on x axis
        RAAN = 0
        arg_p = 0
        theta = m.acos(r[0]/r_mag)
        if r[1]<0: theta = 2*m.pi - theta

    ob = Orbit(p,e,i,RAAN,arg_p,0,sgp) # type: ignore (for e, which is "floating", not float)
    # now figure out the passage at periapsis:
    ob.link_time_and_theta(theta, time) # i swear to god you must work!
    return ob

def orbit_from_lambert(r1:np.ndarray, r2:np.ndarray, start_time:float,
                        end_time:float, sgp:float, short_way:bool = True)->Orbit:
    '''creates an orbit by solving lambert's problem\n
    start_time is the time at r1, end_time is the time at r2. can choose between long and short way.\n
    for getting only the vectors, use lambert_vectors()\n
    start_time also used to set orbit in proper epoch'''
    v1, _ = lambert_vectors(r1,r2,(end_time-start_time),sgp,short_way)
    ob = orbit_from_rv(r1,v1, sgp, start_time)
    return ob

def orbit_from_gauss(observations:list[np.ndarray],
                        times:list[float], 
                        positions:list[np.ndarray],
                        sgp:float)->"Orbit":
    '''Uses Gauss' method to find an orbit from 3 observations, taken at 3 different times.
    you also need to include the positions of the three observations at those 3 given times
    observations assumed to be normal vectors'''
    # TODO UNTESTED!!!!!

    # Basic checks before we start the process:
    if len(observations) != 3 or len(times) != 3 or len(positions) != 3:
        raise ValueError(f"Incorrect number of arguments for Gauss' method"
                            f"({len(observations)}/3) observations"
                            f"({len(times)}/3) times"
                            f"({len(positions)}/3) positions")
    
    
    # unpack the values
    rho1, rho2, rho3 = observations
    t1, t2, t3 = times
    R1, R2, R3 = positions

    #ensure normalised observations
    rho1 = unit(rho1)
    rho2 = unit(rho2)
    rho3 = unit(rho3)

    # calculate intermediates
    tau1 = t1-t2
    tau3 = t3-t2
    tau = t3-t1
    p1 = np.cross(rho2, rho3)
    p2 = np.cross(rho1, rho3)
    p3 = np.cross(rho1, rho2)
    D0 = np.cross(rho1, p1)

    #calculate the D matrix (0 indexed unlike the Curtis variables)
    D = np.outer(np.array([R1,R2,R3]),np.array([p1,p2,p3]))
    
    # if anything is wrong, check D first

    # calculate more intermediates
    A = 1/D0 * (-D[0,1]*tau3/tau + D[1,1] + D[2,1]*tau1/tau)
    B = 1/(6*D0) * (D[0,1]*(tau3**2 - tau**2)*tau3/tau + D[2,1]*(tau**2 - tau1**2)*tau1/tau)
    E = R2.dot(rho2)
    a = -(A**2 + 2*A*E + R2**2)
    b = -2*sgp*B*(A + E)
    c = -sgp**2 * B**2

    # find root
    root = np.roots([1,0,a,0,0,b,0,0,c])[0]
    # in theory there can be multiple roots, but we simply pick the 1st
    r2 = root

    # find slant ranges
    range1 = 1/D0 * (
        (6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*r2**3 + sgp*D[2,0]*(tau**2 - tau1**2)*tau1/tau3) /
        (6*r2**3 + sgp*(tau**2 - tau3**2))
        - D[0,0]
    )
    range2 = A + sgp*B/(r2**3)
    range3 = 1/D0 * (
        (6*(D[0,2]*tau3/tau1 + D[1,2]*tau/tau1)*r2**3 + sgp*D[0,2]*(tau**2 - tau3**2)*tau3/tau1) /
        (6*r2**3 + sgp*(tau**2 - tau1**2))
        - D[2,2]
    )

    # find positions (overwriting r2)
    r1 = R1 + range1*rho1
    r2 = R2 + range2*rho2
    r3 = R3 + range3*rho3

    # find lagrange coefficients
    f1 = 1 - 1/2 * sgp/r2**3 * tau1**2
    g1 = tau1 - 1/6 * sgp/r2**3 * tau1**3
    f3 = 1 - 1/2 * sgp/r2**3 * tau3**2
    g3 = tau3 - 1/6 * sgp/r2**3 * tau3**3

    # find v2
    v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)

    # refine solution using universal anomaly
    chi_l_bound = -np.pi
    chi_u_bound = np.pi # ??? TODO
    error = 1
    while (error > 1e-8):

        old_v2 = v2
        old_r2 = r2

        magr2 = np.linalg.norm(r2)
        a = 1/(2/magr2 - (v2.dot(v2)/sgp))
        vr2 = v2.dot(unit(r2))

        # solve for universal anomaly
        # let's try bisection like in lambert vectors
        def fn(x): return (magr2*vr2/(np.sqrt(sgp)) * x**2 * stumpff_c(x**2/a) +
                            (1-magr2/a) * x**3 * stumpff_s(x**2/a) + magr2*x)
        def fn1(x): return fn(x) - np.sqrt(sgp)*tau1
        def fn3(x): return fn(x) - np.sqrt(sgp)*tau3

        chi1 = root_finder_bisection(fn1, chi_l_bound, chi_u_bound)
        chi3 = root_finder_bisection(fn3, chi_l_bound, chi_u_bound)

        # get new lagrange coefficients
        f1 = 1 - chi1**2/magr2 * stumpff_c(chi1**2/a)
        g1 = tau1 - 1/np.sqrt(sgp) * chi1**3 * stumpff_s(chi1**2/a)
        f3 = 1 - chi3**2/magr2 * stumpff_c(chi3**2/a)
        g3 = tau3 - 1/np.sqrt(sgp) * chi3**3 * stumpff_s(chi3**2/a)

        # get new slant ranges with this (using intermediates)
        c1 = g3/(f1*g3 - f3*g1)
        c3 = -g1/(f1*g3 - f3*g1)
        range1 = 1/D0 * (-D[0,0] + D[1,0]/c1 - c3/c1 * D[2,0])
        range2 = 1/D0 * (-c1*D[0,1] + D[1,1] - c3 * D[2,1])
        range3 = 1/D0 * (-c1/c3*D[0,2] + D[1,2]/c3 - D[2,2])

        # find positions again
        r1 = R1 + range1*rho1
        r2 = R2 + range2*rho2
        r3 = R3 + range3*rho3

        # find v2
        v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)

        # calculate error:
        error = np.linalg.norm(old_v2 - v2) + np.linalg.norm(old_r2 - r2)
        # not the most theoretically sound error but it works (hopefully)
        continue
    # now we return the improved measurements
    return orbit_from_rv(r2, v2, sgp, t2)

def point_to_point(p1:np.ndarray, p2:np.ndarray, radius:float, start_time:float, end_time:float, sgp:float, angular_speed:float, epoch_angle:float)->Orbit:
    '''Create a point to point orbit between two coordinates on a sphere with given radius\n
    points given in elevation/azimuth.\n
    essentially a wrapper of the lambert function'''

    long_shift_1 = epoch_angle + start_time*angular_speed
    long_shift_2 = epoch_angle + end_time*angular_speed
    r1 = elaz_vector(p1[0], p1[1] + long_shift_1, radius)
    r2 = elaz_vector(p2[0], p2[1] + long_shift_2, radius)
    v1,_ = lambert_vectors(r1,r2,end_time-start_time,sgp,True)
    # ensure orbit isn't inside the planet by checking we go upwards:
    if v1.dot(r1) <= 0: raise ValueError("too short time given")

    return orbit_from_rv(r1,v1, sgp, start_time)


# ==== trajectories and intercepts ======
'''
Functions for creating or optimising intercepts between two orbits
'''

def trajectory_optimizer(
        origin:Orbit,
        destination:Orbit,
        start_time:float,
        end_time:float,
        w_insertion:float = 1,
        w_relv:float = 0,
        w_travel_time:float = 0,
        w_intercept_distance:float=0,
        w_intercept_time:float=0
)->tuple[float,float,float,float,float]:
    '''
    Function to optimize the trajectory between two keplerian orbits.
    Uses Nelder-Mead method to find optimum given the optimization weights

    :param origin: Origin orbit
    :type: Orbit
    :param destination: Destination orbit
    :type: Orbit
    :param start_time: Earliest time to leave from origin
    :type: float
    :param end_time: Latest time to arrive at destination
    :type: float
    :param w_insertion: how much the insertion dV is weighted in the optimizer (per km/s)
    :type: float:
    :param w_relv: how much the rendezvous relative velocity is weighted in the optimizer (per km/s)
    :type: float:
    :param w_travel_time: how much the travel time is weighted in the optimizer (per day)
    :type: float:
    :param w_intercept_distance: how much distance from barycenter at intercept is weighted in the optimizer (per day)
    :type: float:
    :param w_intercept_time: how much the time after start_time intercept occurs is weighted in the optimizer (per day)
    :type: float:

    
    :returns insertion dV:
    :returns rendezvouz dV:
    :returns start time:
    :returns end time:
    :returns end distance:
    '''
    # 2D optimization using Nelder-Mead
    # first define optimizer function:
    def F(s:float,t:float)->float: # start + travel time
        if s < start_time: return m.inf
        if s + t > end_time: return m.inf # ensure we're not outside bounds
        r1,v1 = origin.time_to_rv(s)
        r2,v2 = destination.time_to_rv(s+t)
        try:
            vl1,vl2 = lambert_vectors(r1,r2,t,origin.sgp)
        except : return m.inf # doesn't work
        weight = float(
            np.linalg.norm(vl1-v1) * w_insertion +
            np.linalg.norm(vl2-v2) * w_relv + 
            t/DAY * w_travel_time +
            np.linalg.norm(r2)/AU * w_intercept_distance +
            (s+t)/DAY * w_intercept_time
        )
        return weight
    
    # TODO
    
    # define points of interest (apses, nodes, ideal hohmann points, etc.)
    # get points in net, do some optimization, then pick best

    # poi_start = [origin.theta_to_time(x) for x in [0,m.pi, -origin.arg_p, m.pi - origin.arg_p]]
    # poi_end = [origin.theta_to_time(x) for x in [0,m.pi, -origin.arg_p, m.pi - origin.arg_p]]
    


    # find starting point with sampling the range:
    sample_range = np.linspace(start_time,end_time,20)
    ss,ee = np.meshgrid(sample_range,sample_range)
    tt = ee - ss
    FF = np.vectorize(F,otypes=[float])(ss,tt)
    idx = np.unravel_index(FF.argmin(), FF.shape)
    s = ss[idx]
    t = tt[idx]
    dt = sample_range[1]-sample_range[0]

    s_opt,t_opt = nelder_mead_2d(F,np.array([s,t]),-dt/2, 1e-6)

    # compute properties:
    r1,v1 = origin.time_to_rv(s_opt)
    r2,v2 = destination.time_to_rv(s_opt+t_opt)
    vl1,vl2 = lambert_vectors(r1,r2,t_opt,origin.sgp)
    return np.linalg.norm(vl1-v1), np.linalg.norm(vl2-v2), s_opt, s_opt+t_opt, np.linalg.norm(r2) # type: ignore

def porkchop_plot(rv1_fn:Callable[[float], tuple[np.ndarray,np.ndarray]],
                    rv2_fn:Callable[[float], tuple[np.ndarray,np.ndarray]],
                    start_range:list[float], end_range:list[float],
                    sgp:float, short_way:bool = True, rendezvous = True, min_alt=0):
    '''
    Solves lambert's problem for all given starting and ending times. rv1_fn and rv2_fn are functions that return the position and velocity at a given time.
    returns a 2d array of all Dv values, and index of the lowest one.\n
    if rendezvous is true then breaking dv is included, otherwise not\n
    if min_alt is selected, all orbits below that altitude between r1 and r2 are discarded (value inf)\n
    use lambert vector on the winning times to obtain the winning orbit
    '''
    dv_best = m.inf
    idx_best = (0,0)
    array = np.zeros((len(start_range), len(end_range)))
    
    for i in range(len(start_range)):
        start_time = start_range[i]
        r1,v1 = rv1_fn(start_time)
        for j in range(len(end_range)):
            end_time = end_range[j]

            if end_time <= start_time:
                array[i,j] = m.inf
                continue

            r2,v2 = rv2_fn(end_time)
            try:
                vinit, vfin = lambert_vectors(r1, r2, (end_time-start_time), sgp, short_way)
            except (ArithmeticError,ValueError):
                # trajectory failed
                array[i,j] = m.inf
                continue

            # check for minimum altitude
            
            if min_alt != 0 and r1.dot(vinit) < 0: # starting by going down
                # check periapsis (stealing from the orbit_from_rv function):
                #peri = a(1-e) = h^2/sgp * 1/(1+e)
                h_vec = np.cross(r1,vinit) # angular momentum vector
                h = np.linalg.norm(h_vec) # angular momentum
                e_vec = np.cross(vinit,h_vec)/sgp - r1/np.linalg.norm(r1) # eccentricity vector
                e = np.linalg.norm(e_vec) # eccentricty
                peri = h**2/sgp * 1/(1+e)
                if peri < min_alt:
                    array[i,j] = m.inf
                    continue
                # TODO: this doesn't cover going up to apoapsis, then down to periapsis
                # and up again. but that's only on long way trips

            dv1 = np.linalg.norm(vinit - v1)
            dv2 = np.linalg.norm(vfin - v2)
            dv = dv1
            if rendezvous: dv += dv2;

            array[i,j] = dv
            
            # check for best:
            if dv < dv_best:
                dv_best = dv
                idx_best = (i,j)
    
    return array, idx_best

def porkchop_intercept(ob1:Orbit, ob2:Orbit,start_range:list[float], end_range:list[float],
                        short_way:bool = True, rendezvous = True, min_alt=0):
    '''calculates the porkchop plot between two orbits, assumes sgp based on the first orbit
    returns a 2d array of all Dv values, and index of the lowest one.\n
    if rendezvous is true then breaking dv is included, otherwise not\n
    if min_alt is selected, all orbits below that altitude between r1 and r2 are discarded (value inf)\n
    use lambert vector on the winning times to obtain the winning orbit'''
    return porkchop_plot(ob1.time_to_rv,ob2.time_to_rv, start_range,end_range,ob1.sgp,short_way,rendezvous,min_alt)






# ======= misc ========

def propagate(r_0:np.ndarray, v_0:np.ndarray, dt:float, sgp:float, tolerance:float = 1e-9)->tuple[np.ndarray, np.ndarray]:
    '''
    Propagate a r and v vector forward in time by dt\n
    Uses the universal variable method, so does not care about the type of orbit
    (barring maybe degenerate ones)\n
    this is essentially equivalent to from_rv().time_to_rv().
    '''

    # using Curtis' numerical method
    mag_r_0 = np.linalg.norm(r_0) # magnitude of initial r vector
    vr_0 = np.dot(r_0, v_0)/mag_r_0 # initial radial velocity

    alpha = 2/mag_r_0 - v_0.dot(v_0)/sgp # inverse semi-major axis
    # quick almost zero check
    if abs(alpha) < tolerance: alpha = 0 # just for those parabolic cases (it messes up stumpff otherwise i've found)
    if abs(vr_0) < tolerance: vr_0 = 0 # same deal for propagating from apses
    root_mu = m.sqrt(sgp)
    chi = root_mu*dt*abs(alpha) # good first guess
    def F(chi): 
        z = alpha * chi**2
        return (mag_r_0*vr_0/root_mu) * chi**2 * stumpff_c(z) + (1-alpha*mag_r_0) * chi**3 * stumpff_s(z) + mag_r_0*chi - root_mu*dt
    
    def F_prime(chi):
        z = alpha * chi**2
        return (mag_r_0*vr_0/root_mu) * chi * (1-alpha*chi**2 * stumpff_s(z)) + (1-alpha*mag_r_0) * chi**2 * stumpff_c(z) + mag_r_0
    
    chi = root_finder_newton(F, F_prime, chi)
    assert abs(F(chi)) < 1e-5

    # lagrange coefficents using chi
    f = 1 - (chi**2/mag_r_0)*stumpff_c(chi**2 * alpha)
    g = dt - 1/root_mu * chi**3 * stumpff_s(chi**2 * alpha)
    r_1 = f*r_0 + g*v_0
    
    f_dot = root_mu/(mag_r_0 * np.linalg.norm(r_1)) * (chi**3 * alpha * stumpff_s(chi**2 * alpha) - chi)
    g_dot = 1 - chi**2/np.linalg.norm(r_1) * stumpff_c(chi**2 * alpha)
    v_1 = f_dot*r_0 + g_dot*v_0

    return r_1, v_1

def lambert_vectors(r1_vec:np.ndarray, r2_vec:np.ndarray, time:float, sgp:float, short_way:bool = True)->tuple[np.ndarray, np.ndarray]:
    '''Solves lamberts problem of finding an orbit from two position vectors and a time between them\n
    can choose between "short" or "long" way (defaults to short), does not consider solutions of multiple periods.\n
    this version returns only the initial and final velocity vectors,
    use lambert_orbit() to obtain the orbit itself (a combination of this and from_rv)'''

    # required constants:
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    # first the change in anomaly:
    d_theta = m.acos((r1_vec/r1).dot(r2_vec/r2)) # cosine of inner angle
    if not short_way:
        d_theta = 2*m.pi - d_theta # outer angle (i think this is the correct way to implement it)

    # canonise the values (such that sgp = 1)
    # since sgp = 1[DU^3]/[TU^2], we choose TU to be the mean of the two vectors
    DU = r1/2 + r2/2
    # then TU = DU^3/sgp:
    TU = m.sqrt(DU**3 / sgp)

    # canonise the values (should make method more stable):
    r1 /= DU
    r2 /= DU
    time /= TU
    sgp = 1

    A = m.sin(d_theta) * m.sqrt((r1*r2)/(1-m.cos(d_theta)))
    S = stumpff_s
    C = stumpff_c
    y = lambda z: r1 + r2 + A*(z*S(z)-1)/np.sqrt(C(z))
    
    # equation to solve is time*sqrt(mu) = x^3*S(z) + A*sqrt(y(z))
    # with x = sqrt(y(z)/C(z))
    # means:
    chi = lambda z: m.sqrt(y(z)/C(z))
    F = lambda z: chi(z)**3 * S(z) + A * m.sqrt(y(z)) - time*m.sqrt(sgp)

    a= 4*m.pi**2 # upper bound
    b = -4*m.pi**2 # lower bound (expand for hyperbolas)
    while y(b) < 0: b += 0.1 # adjust lower bound (so y is +ve)
    while not m.isfinite(F(a)): a *= 0.9 # adjust upper bound (otherwise it's NaNs)

    z = root_finder_bisection(F,b,a)
    # assert abs(F(z)) < 1e-5

    # f and g_dot are unitless, g is not, having units of [TU]
    f = 1 - y(z)/r1
    g_dot = 1 - y(z)/r2
    g = A*m.sqrt(y(z)/sgp) * TU
    
    v1_vec = (r2_vec - f*r1_vec)/g
    v2_vec = (g_dot*r2_vec - r1_vec)/g
    return v1_vec, v2_vec

def orbit_within_1_precent(ob1:Orbit,ob2:Orbit):
    '''are the orbits within 1% on the 6 elements?
    relative to ob1, will normalize the orbit'''
    ob1.normalize()
    ob2.normalize()

    within = lambda x,y: abs((x-y)/x) < 0.01

    if not within(ob1.p, ob2.p): return False
    if not within(ob1.e, ob2.e): return False
    if not within(ob1.i, ob2.i): return False
    if not within(ob1.RAAN, ob2.RAAN): return False
    if not within(ob1.arg_p, ob2.arg_p): return False
    if not within(ob1.t_p, ob2.t_p): return False
    return True

def plot_orbit(ax,ob:Orbit,time:float=0,trail:float=2*m.pi, ThreeDee:bool=True,hyper_predict:bool=False, max_alt:float=m.inf, **kwargs)->None:
    '''Plot orbit in the given axis, trail determines how far behind the orbit is plotted (defaults to entire orbit)
    hyperbolic orbits will only be plotted up to the current point, if hyper_predict is true, a dashed line will be plotted ahead,
    max_size determines how far out to plot'''

    theta = ob.time_to_theta(time)

    if trail != 2*m.pi:
        # add trail:
        end_theta = theta
        start_theta = theta-trail
    else: # no trail, 
        end_theta = m.pi
        start_theta = -m.pi
    if ob.e >= 1: end_theta = theta

    cross = ob.crosses_altitude(max_alt)
    if cross is None and ob.periapsis > max_alt: return # don't render anything
    elif not cross is None:
        start_theta = bounds(-cross,start_theta,cross)
        end_theta = bounds(-cross,end_theta,cross)
    
    locus = ob.point_locus(start_theta,end_theta)
    if cross is None or abs(theta) < cross:
        point = ob.theta_to_rv(theta)[0]
    else: point = np.array([m.inf])

    if not 'color' in kwargs:
        kwargs['color'] = np.random.random(3)
    if ThreeDee: ax.plot(locus[:,0],locus[:,1],locus[:,2], **kwargs)
    else: ax.plot(locus[:,0],locus[:,1], **kwargs)
        
    kwargs.pop('label',None) # to not duplicate labels
    if np.linalg.norm(point) <= max_alt:
        if ThreeDee: ax.scatter(point[0],point[1],point[2], **kwargs)
        else: ax.scatter(point[0],point[1], **kwargs)
    
    if hyper_predict and ob.e >= 1:
        start_theta = end_theta
        end_theta = cross
        locus = ob.point_locus(start_theta,end_theta) # type:ignore
        if ThreeDee: ax.plot(locus[:,0],locus[:,1],locus[:,2], ls='--', **kwargs)
        else: ax.plot(locus[:,0],locus[:,1], **kwargs)
    return;

