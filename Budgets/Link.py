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
    return 10*np.log(P)


# universe parameters
k = 1.38*10**(-23)          # Boltzmann constant (J/K)
k_db = -23*power_to_Db(1.38)
AU_to_m = 149597870691
r_e = 6378000               # radius (m) of the earth
d_e = 2*r_e                 # diameter (m) of the earth 
step = 100

# S/c antenna parameters
spectrum_utilisation = 1        # no. of bits transmitted per unit frequency (bps/Hz)
T = 13.5             # System noise temperature (K) - depends on frequency
#hpbw = (0.5, 0.5)       # half-power-beam-width (rad, rad) in spherical coordinate system (theta, phi)
eta_P = 0.5        # power  / power into the antenna
eta_trans = 0.18         # Transmitter efficiency (from adsee yr 1)

# G/s antenna parameters
sn = 1.3            # Signal-to-Noise ratio (from requirements)
A_ant = 20.5            # g/s antenna frontal area (m^2)
eta_ant = 0.6           # g/s antenna efficiency (from adsee yr 1)

# Trajectory parameters
t_gs = 0.00001    # time (s) between two consequetive passes over the ground station
tr_time = 1       # time (s) to transmit per pass

# Power subsystem parameters
P_in_db = 250            # Power (w) into the transmitter (from power subsystem)

# Mission parameters
d_max = 3*AU_to_m            # max distance (m) from s/c to g/s
hpbw = (d_e/d_max, d_e/d_max)       # small angle approx
print(hpbw[0]*180/np.pi)

d = np.linspace(20000, d_max, 100000) # distance (m) from s/c to g/s

# Data in
## Payload
data_radar = image_to_bit(3, (10, 10), 3, 10, 0.75) #example
data_payload = data_radar   # plus other things
## Housekeeping
data_thermometer = analog_to_bit(10, 16)    #example
data_housekeeping = data_thermometer    # plus other things
## Total
data_in = data_payload + data_housekeeping

memory_size = data_in*t_gs      # memory size (bits)

tr_max = data_in + memory_size/tr_time      # max telemetry bit-rate (bps)


B = tr_max / spectrum_utilisation      # bandwidth (Hz)

#N0 = k*T            # Noise spectral density (W/Hz)
N0_db = k_db + power_to_Db(T)
#N = N0*B               # Noise power (W)
N_db = N0_db + power_to_Db(B)


#C_req = N*sn            # Required recieved power (W) at groundstation
C_req_db = N_db + power_to_Db(sn)

D = 4*np.pi/(hpbw[0]*hpbw[1])       # Directivity
G = eta_P*D         # Gain
G_db = 400           # for Rosetta


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

plt.plot(np.linspace(0, 10000, 10), np.repeat(P_in_db, 10))
plt.plot(np.linspace(10000, 20000, 10),np.linspace(P_in_db, EIRP_db, 10))
plt.plot(d, EIRP_db - power_to_Db(np.linspace(0.1, 4*np.pi*d_max*d_max, 100000)))
plt.axline((20000, C_req_db), (100000, C_req_db))





plt.show()





