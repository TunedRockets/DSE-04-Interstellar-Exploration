'''
Script for generating a distributions of dV expected for the ISO intercept


'''

from src2.orbit import Orbit, oberth_effect_optimzer, plot_orbit, orbit_from_lambert, orbit_from_rv, orbit_from_ephemeris
from src2.get_ISO import get_ISO
from src2.examples import Earth, Jupiter
from src2.utilities import AU, YEAR, SGP_SUN
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl
mpl.use('TkAgg')




def add_dv_hist(rm, weights, N, PLOT=False)->None:
    '''Adds to the dv histogram for the different weights'''
    np.seterr(all="ignore")
    path = Path(__file__).parent.parent / "data_oberth" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            count = int(lines[0])
            hist = [int(x) for x in lines[1:]]
        

    except:
        hist = [0 for _ in range(100)]
        count = 0


    ISOs = get_ISO() # sample of ISOs
    while len(ISOs) < N:
        ISOs.extend(get_ISO())
    count += len(ISOs)

    for ISO in tqdm(ISOs,desc="studying ISOs"):
        detect_theta = ISO.crosses_altitude(rm*AU)
        if detect_theta is None: continue
        detect_time = ISO.theta_to_time(-detect_theta)
        try:
            # ===============================
            # Oberth optimization at periapsis
            # ===============================
            theta_pe = 0
            tp = origin.theta_to_time(theta_pe)

            rp_vec, vp_vec = origin.theta_to_rv(theta_pe)
            vp_mag = np.linalg.norm(vp_vec)

            min_time = 100

            insert_dv, rdvz_dv, transfer_orbit, st, et = oberth_effect_optimzer(
                ISO,
                rp_vec,
                vp_mag,
                tp,
                min_time,
                max_time,
                optimize_rendezvous=(weights["w_relv"] > 0),
                period=origin.period,
                detect_time=detect_time
            )

            # ===============================
            # Compute required periapsis direction
            # ===============================
            r_req, v_req = transfer_orbit.theta_to_rv(theta_pe)
            v_req_hat = v_req / np.linalg.norm(v_req)

            # current orbit periapsis direction
            r0, v0 = origin.theta_to_rv(theta_pe)
            v0_hat = v0 / np.linalg.norm(v0)

            # ===============================
            # Angle between directions
            # ===============================
            cos_dtheta = np.clip(np.dot(v0_hat, v_req_hat), -1, 1)
            delta = m.acos(cos_dtheta)

            # ===============================
            # Apoapsis velocity
            # ===============================
            r_a = origin.apoapsis
            a = origin.a
            mu = origin.sgp

            v_ap = m.sqrt(mu * (2 / r_a - 1 / a))

            # ===============================
            # Plane change Δv at apoapsis
            # ===============================
            dv_plane = 2 * v_ap * m.sin(delta / 2)

            # ===============================
            # Total insertion Δv
            # ===============================
            insert_dv += dv_plane

        except:
            continue

        insert_dv = round(insert_dv)
        if PLOT:
            # ===============================
            # Reconstruct rotated orbit (after apoapsis burn)
            # ===============================

            # rotation axis from current to required direction
            axis = np.cross(v0_hat, v_req_hat)
            norm = np.linalg.norm(axis)

            if norm < 1e-10:
                axis = np.array([0, 0, 1])  # fallback
            else:
                axis /= norm

            def rotate(vec):
                return (
                        vec * m.cos(delta) +
                        np.cross(axis, vec) * m.sin(delta) +
                        axis * np.dot(axis, vec) * (1 - m.cos(delta))
                )

            # get apoapsis state
            theta_ap = m.pi
            t_ap = origin.theta_to_time(theta_ap)
            r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

            # rotate state
            r_rot = rotate(r_ap)
            v_rot = rotate(v_ap_vec)

            # rebuild orbit
            try:
                origin_rot = orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
                origin_rot.link_time_and_theta(theta_ap, t_ap)
                origin_rot.normalize()
            except:
                continue

            # ==== plotting ====
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # plot original orbit
            plot_orbit(ax, origin, time=detect_time, ThreeDee=True, label="Original")

            # plot rotated orbit
            plot_orbit(ax, origin_rot, time=detect_time, ThreeDee=True, label="Rotated")

            try:
                plot_orbit(ax, transfer_orbit, time=et, ThreeDee=True, label="Transfer", max_alt=(20*AU))
            except:
                pass  # lambert sometimes fails

            # plot ISO, earth and jupiter orbit for context
            plot_orbit(ax, ISO, time=et, ThreeDee=True, label="ISO", max_alt=(20*AU))
            plot_orbit(ax, Earth, time=detect_time, ThreeDee=True, label="Earth")
            plot_orbit(ax, Jupiter, time=detect_time, ThreeDee=True, label="Jupiter")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.axis("scaled")
            textstr = (
                f"ΔV insert: {insert_dv:.2f} km/s\n"
                f"ΔV rendezvous: {rdvz_dv:.2f} km/s\n"
                f"Intercept distance: {np.linalg.norm(transfer_orbit.time_to_rv(et)[0])/AU:.2f} AU\n"
                f"Intercept time: {et/YEAR:.2f} years\n"
            )

            ax.text2D(0.02, 0.98, textstr,
                      transform=ax.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.legend()

            plt.show()
        if insert_dv > 100: continue
        hist[insert_dv] += 1
    
    # Save
    path = Path(__file__).parent.parent / "data_oberth" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    with open(path, "w") as file:
        file.write(str(count) + '\n')
        file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(rm, weights)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    path = Path(__file__).parent.parent / "data_oberth" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    

    with open(path, "r") as file:
        lines = file.readlines()
        count = int(lines[0])
        hist = [int(x) for x in lines[1:]]
    return [x/count for x in hist]






if __name__ == "__main__":

    

    # ==== settings =====

    T = 0 # time window of ISO generation
    # interecept only optimizes for minimum insertion, rendezvous optimises for minimum total dV
    icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    weight = icpt_weights

    detect_distance = 3*AU
    max_time = 40*YEAR


    origin = orbit_from_ephemeris(
        2.61696589776 * AU, #Semi major axis
        0.987573 , #eccentricity
        m.radians(1.303), #Inclination
        m.radians(100.46457166), #Mean longitude
        m.radians(60.0), #Longitude of perihelion
        m.radians(100.464), #Right ascension of ascending node
        SGP_SUN
    )

    
    rm = 3
    add_dv_hist(rm,weight,2000, PLOT=True)

    hist = get_dv_hist(rm, weight)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")

    plt.show()