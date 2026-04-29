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



# universe parameters
k = 1.38*10**(-23)          # Boltzmann constant (J/K)

# S/c antenna parameters
spectrum_utilisation = 1        # no. of bits transmitted per unit frequency (bps/Hz)
T = 500             # System noise temperature (K) - depends on frequency
hpbw = (0.5, 0.5)       # half-power-beam-width (rad, rad) in spherical coordinate system (theta, phi)
eta_P = 0.5        # power  / power into the antenna
eta_trans = 0.18         # Transmitter efficiency (from adsee yr 1)

# G/s antenna parameters
sn = 2.3            # Signal-to-Noise ratio (from requirements)
A_ant = 20.5            # g/s antenna frontal area (m^2)
eta_ant = 0.6           # g/s antenna efficiency (from adsee yr 1)

# Trajectory parameters
t_gs = 1000     # time (s) between two consequetive passes over the ground station
tr_time = 100       # time (s) to transmit per pass

# Power subsystem parameters
P_in = 5            # Power (w) into the transmitter (from power subsystem)



# Data in
## Payload
data_radar = image_to_bit(3, (10, 10), 4, 10, 0.75) #example
data_payload = data_radar   # plus other things
## Housekeeping
data_thermometer = analog_to_bit(10, 16)    #example
data_housekeeping = data_thermometer    # plus other things
## Total
data_in = data_payload + data_housekeeping

memory_size = data_in*t_gs      # memory size (bits)

tr_max = data_in + memory_size/tr_time      # max telemetry bit-rate (bps)


B = tr_max / spectrum_utilisation      # bandwidth (Hz)

N0 = k*T            # Noise spectral density (W/Hz)
N = N0*B               # Noise power (W)


C_req = N*sn            # Required recieved power (W) at groundstation

D = 4*np.pi/(hpbw[0]*hpbw[1])       # Directivity
G = eta_P*D         # Gain

EIRP = P_in*eta_trans*G
d = 10000000            # distance (m) from s/c to g/s
Wf = EIRP/(4*np.pi*d*d)     # Power flux density (W/m^2)

C_real = Wf*A_ant*eta_ant/2     # Actually recieved power (W) at the groundstation (worst case)
print(C_real - C_req)





