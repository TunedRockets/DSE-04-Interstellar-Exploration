import numpy as np
import matplotlib.pyplot as plt

def image_to_bit(fps, res, channels, bits_per_channel, CF):
    """ Converts image specs to bit-rate (bps) 
        Inputs:
            - float: frames-per-second
            - (int, int): resolution
            - int: no. of channels
            - int: bits_per_channel
            - float: compression factor
    """
    return fps*res[0]*res[1]*channels*bits_per_channel/CF

def QE(nbps):
    """ Calculates quantisation error as percentage
        Input:
            - int: no of bits per signal
    """
    return 100/2**(nbps+1)

def analog_to_bit(f_signal, nbps, N=2.2):
    """ Converts analog sampler specs to bit-rate (bps)
        Inputs:
            - float: signal frequency [Hz]
            - int: no. of bits per signal
            - float: Nyquist number (from ADSEE yr 1)
    """
    print(f"{QE(nbps)} %")
    return f_signal*nbps*N

def power_to_Db(P):
    return 10*np.log10(P)


# universe parameters
k = 1.38*10**(-23)          # Boltzmann constant (J/K)
k_db = -230 + power_to_Db(1.38)
print(k_db)
AU_to_m = 149597870691
r_e = 6378000               # radius (m) of the earth
d_e = 2*r_e                 # diameter (m) of the earth 
step = 100

# S/c antenna parameters
spectrum_utilisation = 0.1        # no80. of bits transmitted per unit frequency (bps/Hz)
T = 20             # System noise temperature (K) - depends on frequency
print(power_to_Db(T))
#hpbw = (0.5, 0.5)       # half-power-beam-width (rad, rad) in spherical coordinate system (theta, phi)
eta_P = 0.5        # power  / power into the antenna
eta_trans = 0.18         # Transmitter efficiency (from adsee yr 1)

# G/s antenna parameters
sn_db = -2.7            # Signal-to-Noise ratio (from requirements)
A_ant = np.pi*20            # g/s antenna frontal area (m^2)
eta_ant = 0.6           # g/s antenna efficiency (from adsee yr 1)
G_ant_db = 68.3

# Trajectory parameters
t_gs = 0.00001    # time (s) between two consequetive passes over the ground station
tr_time = 1       # time (s) to transmit per pass

# Power subsystem parameters
P_in_db = power_to_Db(22)            # Power (w) into the transmitter (from power subsystem)

# Mission parameters
d_max = 3*AU_to_m            # max distance (m) from s/c to g/s
hpbw = (d_e/d_max, d_e/d_max)       # small angle approx
print(hpbw[0]*180/np.pi)

d = np.linspace(1, d_max, 100000) # distance (m) from s/c to g/s

# Data in
## Payload
data_radar = image_to_bit(3, (10, 10), 3, 8, 0.75) #example
data_payload = data_radar   # plus other things
## Housekeeping
data_thermometer = analog_to_bit(10, 16)    #example
data_housekeeping = data_thermometer    # plus other things
## Total
data_in = data_payload + data_housekeeping

memory_size = data_in*t_gs      # memory size (bits)

tr_max = data_in + memory_size/tr_time      # max telemetry bit-rate (bps)
print(f"{tr_max} max bitrate")
tr_max = 160


B = tr_max / spectrum_utilisation      # bandwidth (Hz)
print(B)

#N0 = k*T            # Noise spectral density (W/Hz)
N0_db = k_db + power_to_Db(T)
print(f"{N0_db} N0_db")
#N = N0*B               # Noise power (W)
N_db = N0_db + power_to_Db(B)


#C_req = N*sn            # Required recieved power (W) at groundstation
C_req_db = N_db + sn_db
#C_req_db = -181.4

D = 4*np.pi/(hpbw[0]*hpbw[1])       # Directivity
G = eta_P*D         # Gain
G_db = 40           # for Rosetta


#EIRP = P_in*eta_trans*G
EIRP_db = P_in_db + power_to_Db(eta_trans) + G_db
#Wf_min = EIRP/(4*np.pi*d_max*d_max)     # Power flux density (W/m^2)
#Wf = EIRP/(4*np.pi*d*d)     # Power flux density (W/m^2)
Wf_db = EIRP_db - power_to_Db(4*np.pi*d*d)

#C_real_min = Wf[-1]*A_ant*eta_ant/2     # Actually recieved power (W) at the groundstation (worst case)
C_real_min_db = Wf_db[-1] - power_to_Db(A_ant*eta_ant/2)

#C_real = Wf*A_ant*eta_ant/2     # Actually recieved power (W) at the groundstation (worst case)
C_real_db = Wf_db - power_to_Db(A_ant*eta_ant/2)
print(C_real_db[-1] - C_req_db)

print(P_in_db, N_db, EIRP_db, C_real_db[-1], C_req_db)
# --- Plotting ---
plt.figure(figsize=(10, 6))

plt.plot(np.linspace(-2e10, -1e10, 10), np.repeat(P_in_db, 10))
plt.plot(np.linspace(-1e10, 0, 10),np.linspace(P_in_db, EIRP_db, 10))
plt.plot(d, EIRP_db - power_to_Db(np.linspace(0, 4*np.pi*d_max*d_max, 100000)))
plt.plot(np.linspace(d_max, d_max+1e10, 10), np.repeat(C_real_min_db + G_ant_db, 10))
plt.axline((20000, C_req_db), (100000, C_req_db))





plt.show()





