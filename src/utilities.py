'''
some utility functions and constants that are useful when calculating astrodynamical things
By: Johannes Nilsson
'''
import numpy as np
import math as m
from typing import Callable
import matplotlib.pyplot as plt

# === constants =============================

# EPOCH_SIDERIAL = 1.749_333_40 # [rad] LST at Greenwich at unix 0 (00:00 jan 1st 1970)
SIDERIAL_DAY = 1.002_737_811_911_354_48*2*m.pi # [rad] rotation of greenwich meridian in one UT day
# specifically the rotation speed at the year 2000,
SIDERIAL_2000_ADJUSTMENT = 2*m.pi*(0.779_057_273_2640) # adjustment for calculating local siderial time
EARTH_ANGULAR_SPEED = SIDERIAL_DAY / 86_400 # [rad/s] rotation speed of earth
FT_TO_M = 0.3048 # [ft/m] standard conversion from BMW example values
EQ_RAD_EARTH = 6378.145 # [km] mean equatorial radius
E_EARTH = 0.08182 # [-] eccentricity of the ellipsoid earth
SGP_EARTH = 3.986012e5 # [km^3/s^2] mass of earth times gravitational constant

SGP_SUN = 1.32712e11 # [km^3/s^2]
AU = 149_597_870.7 # [km]
DAY = 60*60*24 # [s]
YEAR = DAY*365.25 # [s] (Julian Year)

# canonical units (from BMW, a set of units that make values around 1 for normal orbits (+- some magnitude))
# based around the earth radius and mass
CANON_DU = EQ_RAD_EARTH # [km] canonical distance unit (earth)
CANON_TU = 806.8118744 # [s] canonical time unit (earth)
CANON_SU = CANON_DU/CANON_TU # [km/s] canonical speed unit (earth)


# === mathematical functions ========

def stumpff_s(z:float)->float:
    ''' Stumpff sine function, also known as c_3,
    equivalent to the infinite series:
    (-z)^n / (2n + 3)! for n->infinity'''

    if z==0:
        return 1/6
    elif z > 0:
        z_sqrt = m.sqrt(z)
        return (z_sqrt - m.sin(z_sqrt))/(z_sqrt**3)
    else:
        z_sqrt = m.sqrt(-z)
        return (-z_sqrt + m.sinh(z_sqrt))/(z_sqrt**3)

def stumpff_c(z:float)->float:
    ''' Stumpff cosine functionalso known as c_2,
    equivalent to the infinite series:
    (-z)^n / (2n + 2)! for n->infinity'''

    if z==0:
        return 1/2
    elif z > 0:
        return (1-m.cos(m.sqrt(z)))/z
    else:
        return (-1 + m.cosh(m.sqrt(-z)))/(-z)

def root_finder_bisection(f:Callable, lower:float, upper:float, tolerance:float = 1e-8)->float:
    '''takes a univariate function and finds the root of that function
    through recursive bisection.
    converges on a root between bounds, provided bounds are of different sign'''

    if not ( f(lower) * f(upper) < 0): # check the initial interval contains a root
        raise ValueError("bounds have same sign")           
    while 0.5*np.abs(upper-lower) > tolerance:  # check that we're not converged
        middle = (lower + upper)/2                  # midpoint of current interval
        if f(lower) * f(middle) < 0:           # select which 1/2 interval to continue with
            upper = middle
        else:
            lower = middle
    return middle

def root_finder_newton(f:Callable[[float],float], df:Callable[[float],float],x0:float, max_iter:int = 100, precision=1e-6)->float:
    '''runs newton's method of root finding, will throw an arithmetic error on divergence'''

    for i in range(max_iter):
        fx = f(x0)
        if abs(fx) < precision: return x0
        dx = fx/df(x0)
        x0 = x0 - dx
    else: raise ArithmeticError("Newton's method failed to converge")

def nelder_mead_2d(f:Callable[[float,float],float],x0:np.ndarray, x0_size:float, precision:float = 1e-6, max_iter:int=500, allow_nonconvergence:bool=False)->tuple[float,float]:
    '''Implementation of the Nelder Mead optimization algorithm,
    uses default values for the coefficients,
    (minimizes the value)

    :param f: function to minimize
    :type f: Callable[[float,float],float]
    :param x0: initial point
    :type x0: np.ndarray
    :param x0_size: initial step size
    :type x0_size: float
    :param precision: standard deviation required to terminate, defaults to 1e-6
    :type precision: float, optional
    :param max_iter: maximum allowed iterations, each iteration samples the function a maximum of 3 times, defaults to 500
    :type max_iter: int, optional
    :param allow_nonconvergence: if this is true, the function will return current value on reaching maximum iterations, even
    if it hasn't converged, defaults to False
    :type allow_nonconvergence: bool, optional
    :return: coordinates of minimum value of f
    :rtype: tuple[float,float]
    '''
    a = 1
    b = 0.5
    c = 2
    d = 0.5


    p1 = [0,x0]
    p2 = [0,x0 + np.array([x0_size,0])]
    p3 = [0, x0 + np.array([0,x0_size])]

    p1[0] = f(*p1[1])
    p2[0] = f(*p2[1])
    p3[0] = f(*p3[1])
    # arrr = np.column_stack((p1[1],p2[1],p3[1],p1[1]))
    # plt.plot(arrr[0], arrr[1])
    avg_point = lambda p1,p2,p3: ((p1[1][0] + p2[1][0] + p3[1][0])/3, (p1[1][1] + p2[1][1] + p3[1][1])/3)

    for _ in range(max_iter):
            
        # ordering (lowest first):
        if p1[0] > p2[0]: p1,p2 = p2,p1
        if p2[0] > p3[0]: p2,p3 = p3,p2
        if p1[0] > p2[0]: p1,p2 = p2,p1

        # termination (based on ssd of function values):
        m = (p1[0] + p2[0] + p3[0])/3
        var = (((p1[0]-m)**2 + (p2[0]-m)**2 + (p3[0]-m)**2)/2)
        if var < precision**2:
            return avg_point(p1,p2,p3)

        # centroid:
        cent = 0.5*(p1[1] + p2[1])

        # transform:
        reflect_p = cent + a*(cent - p3[1])
        reflect = [f(*reflect_p), reflect_p]
        if p1[0] <= reflect[0] < p2[0]:
            p3 = reflect
            continue

        elif reflect[0] < p1[0]:
            expand_p = cent + c*(reflect_p - cent)
            expand = [f(*expand_p), expand_p]
            if expand[0] < reflect[0]:
                p3 = expand
                continue
            else:
                p3 = reflect
                continue
        elif reflect[0] < p2[0]:
            if reflect[0] < p3[0]:
                contract_p = cent + b*(reflect_p - cent)
            else:
                contract_p = cent + b*(p3[1] - cent)
            contract = [f(*contract_p), contract_p]
            if contract[0] < p3[0]:
                p3 = contract
                continue
        else:
            # shrink:
            p3_p = p1[1] + d*(p3[1]-p1[1])
            p3 = [f(*p3_p), p3_p]
            p2_p = p1[1] + d*(p2[1]-p1[1])
            p2 = [f(*p2_p), p2_p]
            continue
    else:
        if allow_nonconvergence: return avg_point(p1,p2,p3)
        else: raise ArithmeticError("Nelder-mead failed to converge")

def bounds(lower, value, upper):
    '''alias of min(upper, max(lower, value))\n
    works using numpy minimum so can work on arrays'''
    return np.minimum(upper, np.maximum(lower, value))

def inside_modulo_bounds(lower, value, upper, modulo):
    '''
    Moves value into the given bounds by moving wth the modulo,
    if multiple bounds work, will return each instance as a list
    if it cannot be moved into the region, will return empty list
    '''
    
    # move value below lower:
    while value-modulo > lower: value -= modulo
    # go through to add to list
    l = []
    while value <= upper:
        if lower < value < upper: l.append(value)
        value += modulo
    return l


def lerp(a,b,r, clamped=False):
    '''lerps between value a and value b,
    r is the fraction between 0 and 1 that is lerped to\n
    uses np.multiply.outer(), which means that if r is a list, the output is a
    list of vectors each lerped by that amount\n
    if clamped is true, it will force r to be between 0 and 1'''
    if clamped: r = max(0,min(1,r))
    return a - np.multiply.outer(r,a) + np.multiply.outer(r,b)

def spherical_pythagoras(p1,p2):
    '''finds spherical distance between two coordinates on a sphere.
    NOTE: this only applies to great circles, so won't work with coordinates
    i.e. do cos(c) = cos(a)cos(b) rather than c^2 = a^2 + b^2'''
    return np.arccos(np.cos(p1[0]-p2[0])*np.cos(p1[1]-p2[1]))

def haversine_formula(p1,p2):
    '''calculates the great circle (angular) distance between two points
    given in latitude and longitude. equivalent to turning the coordinates into vectors and dotting the result (keep in mind if already delaing with vectors)'''

    h = (1-np.cos(p1[0]-p2[0]) + np.cos(p1[0]) * np.cos(p2[0]) * (1 - np.cos(p1[1]-p2[1])))
    d = 2 * np.arcsin(np.sqrt(h)) # may be at risk of FPE according to literature?
    return d

def distance_formula(p1, bearing, angular_distance):
    '''calculates the endpoint reaced by starting at p1 and travelling a set distance along a bearing
    on a sphere. compare to the haversine_formula, which does the opposite'''

    lat = np.arcsin(np.sin(p1[0])*np.cos(angular_distance) + np.cos(p1[0])*np.sin(angular_distance)*np.cos(bearing))
    lon = p1[1] + np.arctan2((np.sin(bearing)*np.sin(angular_distance)*np.cos(p1[0])),(np.cos(angular_distance) - np.sin(p1[0])*np.sin(lat)))
    return np.array([lat,lon])

def coordinate_linspace(p1:np.ndarray,p2:np.ndarray,num:int, mercator:bool=True, endpoint:bool=True)->np.ndarray:
    '''Generates equadistant points in between p1 and p2 either via mercator or great circles'''
    # turns out this is useless becuase doing it with vectors already is better...
    if mercator:
        return np.linspace(p1,p2,num, endpoint) # default behaviour
    # else:
    # we slerp that bad boy:
    ff = np.linspace(0,1,num,endpoint)
    v1 = elaz_vector(p1[0],p1[1])
    v2 = elaz_vector(p2[0],p2[1])
    d = np.arccos(v1.dot(v2))
    a = np.sin((1-ff)*d)/np.sin(d)
    b = np.sin(ff*d)/np.sin(d)
    vv = np.outer(a,v1) + np.outer(b,v2) # array of vectors evenly spaced out
    Az = np.arctan2(vv[:,1], vv[:,0])
    El = np.pi/2 - np.arctan2(np.sqrt(np.pow(vv[:,1],2)+np.pow(vv[:,0],2)),vv[:,2])
    return np.column_stack((El,Az))

def slerp_linspace(v1,v2,num:int,endpoint:bool=True)->np.ndarray:
    '''create a slerped linspace between two vectors'''
    ff = np.linspace(0,1,num,endpoint)
    d = np.arccos(v1.dot(v2)) # think this only works with unit vectors, otherwise you get a squaring thing
    a = np.sin((1-ff)*d)/np.sin(d)
    b = np.sin(ff*d)/np.sin(d)
    vv = np.outer(a,v1) + np.outer(b,v2) # array of vectors evenly slerped
    return vv


# === Linear algebra =======

def unit(x:np.ndarray)->np.ndarray:
    '''creates a normal vector'''
    return x/np.linalg.norm(x)

def unit_array(xx:np.ndarray)->np.ndarray:
    '''normalizes array of vectors'''
    return xx/np.linalg.norm(xx, axis=1,keepdims=True)

def elaz_vector(elevation, azimuth, range = 1.0)->np.ndarray:
    '''creates unit vector from elevation and azimuth angles,
    azimuth is CCW from x axis, elevation is from -pi/2 to +pi/2\n
    optional range value for full el-az-r coordinate transform'''

    return np.array([np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)])*range

def vector_elazr(r):
    '''returns elevation, azimuth, and range for a vector\n
    inverse of elaz_vector()'''
    Az = np.arctan2(r[1], r[0])
    El = np.pi/2 - np.arctan2(np.sqrt(r[0]*r[0] + r[1]*r[1]),r[2])
    z = np.linalg.norm(r)
    return El, Az, z

def array_elazr(rr)->np.ndarray:
    '''applies vector_elazr to each vector in the array'''
    Az = np.arctan2(rr[:,1], rr[:,0])
    El = np.pi/2 - np.arctan2(np.sqrt(np.pow(rr[:,1],2)+np.pow(rr[:,0],2)),rr[:,2])
    z = np.linalg.norm(rr, axis=1,keepdims=True)
    return np.column_stack((El,Az, z))

def elazr_array(cc:np.ndarray)->np.ndarray:
    '''elazr_vector over entire array'''

    if cc.shape[1] == 2: # no range, put in range
        cc = np.hstack((cc, np.ones((cc.shape[0],1))))

    x = np.cos(cc[:,0])*np.cos(cc[:,1]) * cc[:,2]
    y = np.cos(cc[:,0])*np.sin(cc[:,1]) * cc[:,2]
    z = np.sin(cc[:,0]) * cc[:,2]
    return np.column_stack((x,y,z))


def rot1(angle:float)->np.ndarray:
    '''creates a matrix to rotate a set angle around the first ("x") axis (+ccw)'''
    return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])

def rot2(angle:float)->np.ndarray:
    '''creates a matrix to rotate a set angle around the second ("y") axis (+ccw)'''
    return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

def rot3(angle:float)->np.ndarray:
    '''creates a matrix to rotate a set angle around the third ("z") axis (+ccw)'''
    return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [ 0, 0, 1]
            ])

def rot_unit(pivot:np.ndarray, angle:float|np.ndarray)->np.ndarray:
    '''creates rotation matrix around arbitrary pivot'''

    raise NotImplementedError("FUNCTION DOES NOT WORK FOR SOME REASON")
    #premake variables:
    pivot = unit(pivot)
    x = pivot[0]
    y = pivot[1]
    z = pivot[2]
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array([
        [x*x*(1-c) + c,   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [x*y*(1-c) + z*c, y*y*(1-c) + c,   y*z*(1-c) - x*s],
        [x*z*(1-c) - y*s, y*z*(1-c) + x*s, z*z*(1-c) + c]
    ])

def rodrigues_rot(v:np.ndarray, pivot:np.ndarray, angle:float)->np.ndarray:
    '''rotates one vector around another using Rodrigues' formula, follows 
    right hand rule/CCW convention
    see also: rot_unit'''

    # normalise pivot:
    pivot = unit(pivot)
    return v*np.cos(angle) + (np.cross(pivot,v))*np.sin(angle) + pivot*(pivot.dot(v))*(1-np.cos(angle))

def trans_mat(A_from:np.ndarray = np.eye(3), B_to:np.ndarray = np.eye(3))->np.ndarray:
    '''Creates a transform from one coordinate system to another\n
    I.e. if you have a vector v defined in the A coordinate system, but want it expressed
    in the B coordinate system, then that would be Q@v where Q = transmat()\n
    both matricies must be orthogonal bases of their coordinate systems,
    so A = [i, j, k] where i is the first basis vector etc.\n
    either argument can be left empty, in which case the I matrix will be used instead,
    so trans_mat(A)@v takes a vector v in coordinate system A and returns it in the global
    coordinate system.'''

    # Check for orthonormalcy of matrices:
    if not np.linalg.norm(A_from.T@A_from  - np.eye(3)) < 1e-9:
        raise ValueError("Matrix A is not orthonormal")
    if not np.linalg.norm(B_to.T@B_to  - np.eye(3)) < 1e-9:
        raise ValueError("Matrix B is not orthonormal")

    # create the matrix:
    return B_to.T@A_from

def camera_hom_mat(front:np.ndarray, up:np.ndarray, pos:np.ndarray, F:float)->np.ndarray:
    '''Creates a camera transformation matrix including translation,
    front and up are used to define the camera rotation (up does not necessarily need 
    to be orthogonal with forward),
    pos is the camera position, F is the focal length of the camera\n
    The matrix expects 3D homogeneous coordinates (so 4 entries),
    and returns 2D homogeneous vector in screen space (so 3 entries) with y-down x-right, origin is camera\n
    The scaling factor of the homogeneous vector is going to be the z/w in camera space
    (where w is the scaling factor of the input vector),
    use this for culling and stuff'''

    C = np.array([ # camera matrix, standard for hom coordinates
        [F,0,0,0],
        [0,F,0,0],
        [0,0,1,0]
    ])


    # for the camera to make sense, we need the z-axis to point forward, and the x-axis to point right
    # (with y down)
    # to get a screen coordinate of (0,0) top left, (1,0) top right etc.
    # like this:
    # 0 ---- > x
    # |
    # |
    # |
    # \/
    # y
    # that allows us to easily implement the drawing as that's what for example PyGame expects

    # we make sure y is orthogonal to z
    z = unit(front)
    y = unit(-(up - (up.dot(z))*z)) # making y orthogonal to z and in right direction
    x = np.cross(y,z) # fill out the triad
    R = np.row_stack((x,y,z)) # create the transformation (rotation) matrix

    # in principle, the value of the combined translation/rotation/camera projection is
    # C @ R_h @ T_h 
    # where R_h and T_h are transformation and translation in hom coordinates respectively.
    # this can be simplified to the combined 4x4 trasnformation/rotation:
    # [R R@t] with an extra row below of [0 0 0 1]
    # so, this means
    t = np.array([R@(-pos)]).T  # -pos since we need to take away the camera position
    # and make it a 3D column vector
    T = np.vstack((np.hstack((R, t)),np.array([0,0,0,1]))) 
    Q = C@T
    return Q

def hom_vector(v:np.ndarray)->np.ndarray:
    '''creates a homogeneous vector from a vector\n
    (it adds a 1 to the vector)\n
    remember that to get back to non homogeneous coordinates you divide by the final value'''
    return np.hstack((v,[1]))

def hom_v_matmul_3D(M:np.ndarray, v:np.ndarray)->np.ndarray:
    '''optimized 3D(hom) matrix times 3D vector\n
    simple testing says it's 4x faster'''

    return M@np.array([v[0],v[1],v[2],1])

def from_hom_vector(v:np.ndarray)->np.ndarray:
    '''Creates a non-homogeneous vector from a homogeneous one\n
    (by dividing all coordinates by the last entry)\n
    due to the nature of homogeneous vectors it is one dimension less'''
    return v[0:-1] / v[-1]

def descale_hom_vector(v:np.ndarray)->np.ndarray:
    '''Creates a non-homogeneous vector from a homogeneous one, but keeps the scaling factor'''
    w = v/v[-1]
    w[-1] *= v[-1]
    return w

# === keplers equation utils and solvers =====

def mean_2_true(M:float, e:float)->float:
    '''mean anomaly to true anomaly\n
    automatically gives eccentric, hyperbolic, or parabolic depending on e\n
    uses an underlying newton iterator for the M->E step. precision governes
    how precise the conversion is, and iterations is how long it goes on for at maximum'''

    if e == 1:
        z = np.cbrt(3*M + np.sqrt(1+(3*M)**2))
        theta = 2*np.arctan(z-1/z) # parabolic true anomaly
    elif e < 1:
        # newton iterate to find M -> E
        def Efn(E): return E-e*np.sin(E)-M
        def Edf(E): return 1-e*np.cos(E)

        E = root_finder_newton(Efn, Edf, M)
        theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2)) # eccentric true anomaly
    else:
        # newton iterate to find M -> F
        def Ffn(F): return e*np.sinh(F)-F-M
        def Fdf(F): return e*np.cosh(F)-1

        F = root_finder_newton(Ffn,Fdf,M)
        theta = 2*np.arctan(np.sqrt((1+e)/(e-1))*np.tanh(F/2)) # hyperbolic true anomaly
    return theta
    
def true_2_mean(theta:float, e:float)->float:
    '''true anomaly to mean anomaly\n
    automatically gives eccentric, hyperbolic, or parabolic depending on e'''

    if e < 1:
        E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta/2))
        M = E - e*np.sin(E) # elliptical mean anomaly
    elif e == 1:
        M = 0.5*np.tan(theta/2)+(1/6)*np.tan(theta/2)**3 # parabolic mean anomaly
    else:
        F = 2*np.arctanh(np.sqrt((e-1)/(e+1))*np.tan(theta/2)) # tan(theta) isn't a typo
        M = e*np.sinh(F) - F # hyperbolic mean anomaly
    return M

def time_2_true(t:float,e:float,h:float,sgp:float)->float:
    '''time to true anomaly via the universal variable method'''
    
    # time -> chi -> true
    rp = h*h/(sgp*(1+e))
    alpha = (sgp*(1-e*e))/(h*h) # 1/a
    S = stumpff_s
    C = stumpff_c
    # sqrt(sgp) * t = (1-rp*alpha)chi^3 S(chi^2*alpha) + rp*chi (after periapsis)
    F = lambda chi: (1-rp*alpha)*chi**3 * S(chi**2*alpha) + rp*chi - m.sqrt(sgp)*t
    dF = lambda chi: (1-rp*alpha)*chi**2 * C(chi**2*alpha) + rp
    # find root:
    chi0 = m.sqrt(sgp) * t * abs(alpha)
    chi = root_finder_newton(F,dF,chi0)

    if e == 1: # parabolic:
        return 2 * m.atan(m.sqrt(sgp)*chi/h)
    elif e < 1: # elliptic
        E = chi * m.sqrt(alpha)
        return 2*m.atan(m.sqrt((1+e)/(1-e)) * m.tan(E/2))
    else: # hyperbolic
        Eh = chi * m.sqrt(-alpha)
        return 2*m.atan(
            m.sqrt((e+1)/(e-1)) * m.tanh(Eh/2)
        )


def true_2_time(theta:float, e:float, h:float, sgp:float)->float:
    '''true anomaly to time using universal variable method'''
    alpha = (sgp*(1-e*e))/(h*h)
    rp = h*h/(sgp*(1+e))
    S = stumpff_s

    if e == 1:
        chi = h/(m.sqrt(sgp))*m.tan(theta/2)
    elif e < 1:
        E = 2 * m.tan(m.sqrt((1-e)/(1+e)) * m.tan(theta/2))
        chi = E/(m.sqrt(alpha))
    else:
        Eh = 2 * m.atanh(m.sqrt((e-1)/(e+1)) * m.tan(theta/2))
        chi = Eh/(m.sqrt(-alpha))
    
    time = ((1-rp*alpha)*chi**3 * S(chi**2*alpha) + rp*chi)/(m.sqrt(sgp))
    return time
    

def main():
    print(unit(np.array([5,7,-2])))


if __name__ == "__main__":
    main()