'''
Script for generating a distributions of dV expected for the ISO intercept


'''

from src.orbit import Orbit, trajectory_optimizer, plot_orbit, orbit_from_lambert, orbit_from_rv, orbit_from_ephemeris
from src.get_ISO import get_ISO
from src.examples import Earth, Jupiter
from src.utilities import AU, YEAR, SGP_SUN
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
    path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    
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
            # --- plane alignment ---
            theta_ap = m.pi
            r_ap, _ = origin.theta_to_rv(theta_ap)
            r_hat = r_ap / np.linalg.norm(r_ap)

            h1 = origin.h_vec
            h2 = ISO.h_vec
            h2_proj = h2 - np.dot(h2, r_hat) * r_hat

            cos_dtheta = np.dot(h1, h2_proj) / (np.linalg.norm(h1) * np.linalg.norm(h2_proj))
            cos_dtheta = np.clip(cos_dtheta, -1, 1)
            delta_plane = m.acos(cos_dtheta)

            # aphelion velocity
            r_a = origin.apoapsis
            a = origin.a
            mu = origin.sgp
            v_ap = m.sqrt(mu * (2 / r_a - 1 / a))

            plane_dv = 2 * v_ap * m.sin(delta_plane / 2)

            # rotation axis
            axis = np.cross(h1, h2_proj)
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                axis = np.array([0, 0, 1])
            else:
                axis /= norm

            def rotate(vec):
                return (vec * m.cos(delta_plane) +
                        np.cross(axis, vec) * m.sin(delta_plane) +
                        axis * np.dot(axis, vec) * (1 - m.cos(delta_plane)))

            # rotate orbit at aphelion
            theta_ap = m.pi
            t_ap = origin.theta_to_time(theta_ap)
            r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

            r_rot = rotate(r_ap)
            v_rot = rotate(v_ap_vec)

            origin_rot = orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
            origin_rot.link_time_and_theta(theta_ap, t_ap)
            origin_rot.normalize()

            # --- trajectory optimization ---
            insert_dv, rdvz_dv, st, et, er = trajectory_optimizer(
                origin_rot,
                ISO,
                detect_time,
                detect_time + max_time,
                **weights
            )

            insert_dv += plane_dv
        except: continue
        insert_dv = round(insert_dv)
        if insert_dv > 100: continue
        elif PLOT:
            # ==== plotting ====
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # plot original orbit
            plot_orbit(ax, origin, time=detect_time, ThreeDee=True, label="Original")

            # plot rotated orbit
            plot_orbit(ax, origin_rot, time=detect_time, ThreeDee=True, label="Rotated")

            # ---- plot transfer arc ----
            r1, v1 = origin_rot.time_to_rv(st)
            r2, v2 = ISO.time_to_rv(et)

            try:
                transfer_orbit = orbit_from_lambert(r1, r2, st, et, origin.sgp)
                plot_orbit(ax, transfer_orbit, time=et, ThreeDee=True, label="Transfer", max_alt=(40*AU))
            except:
                pass  # lambert sometimes fails

            # plot ISO, earth and jupiter orbit for context
            plot_orbit(ax, ISO, time=et, ThreeDee=True, label="ISO", max_alt=(40*AU))
            plot_orbit(ax, Earth, time=detect_time, ThreeDee=True, label="Earth")
            plot_orbit(ax, Jupiter, time=detect_time, ThreeDee=True, label="Jupiter")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.axis("scaled")
            ax.legend()

            plt.show()
        hist[insert_dv] += 1
    
    # Save
    # path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    # with open(path, "w") as file:
    #     file.write(str(count) + '\n')
    #     file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(rm, weights)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    path = Path(__file__).parent / "data" / f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}"
    

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
    max_time = 20*YEAR


    origin = orbit_from_ephemeris(
        2.61696589776 * AU, #Semi major axis
        0.987573 , #eccentricity
        m.radians(1.303), #Inclination
        m.radians(100.46457166), #Mean longitude
        m.radians(60.0), #Longitude of perihelion
        m.radians(100.464), #Right ascension of ascending node
        SGP_SUN
    )

    
    rm = 4
    add_dv_hist(rm,weight,2000, PLOT=True)

    hist = get_dv_hist(rm, weight)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.show()

    raise NotImplementedError()
    


    # ==== generate scenario ======
    ISOs = get_ISO() # sample of ISOs
    while len(ISOs) < 400:
        ISOs.extend(get_ISO())
    print(f"Number of generated ISOs: {len(ISOs)}")


    # === inspect ISOs for their properties ====
    ISO_stats = []
    invalid_count = 0
    for ISO in tqdm(ISOs,desc="Studying each ISO"):
        detect_theta = ISO.crosses_altitude(detect_distance)
        if detect_theta is None: continue
        detect_time = ISO.theta_to_time(-detect_theta)
        try:
            # --- plane alignment ---
            h1 = origin.h_vec
            h2 = ISO.h_vec

            cos_dtheta = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
            cos_dtheta = np.clip(cos_dtheta, -1, 1)
            delta_plane = m.acos(cos_dtheta)

            # aphelion velocity
            r_a = origin.apoapsis
            a = origin.a
            mu = origin.sgp
            v_ap = m.sqrt(mu * (2 / r_a - 1 / a))

            plane_dv = 2 * v_ap * m.sin(delta_plane / 2)

            # rotation axis
            axis = np.cross(h1, h2)
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                axis = np.array([0, 0, 1])
            else:
                axis /= norm


            def rotate(vec):
                return (vec * m.cos(delta_plane) +
                        np.cross(axis, vec) * m.sin(delta_plane) +
                        axis * np.dot(axis, vec) * (1 - m.cos(delta_plane)))


            # rotate orbit at aphelion
            theta_ap = m.pi
            t_ap = origin.theta_to_time(theta_ap)
            r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

            r_rot = rotate(r_ap)
            v_rot = rotate(v_ap_vec)

            origin_rot = Orbit.orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
            origin_rot.link_time_and_theta(theta_ap, t_ap)
            origin_rot.normalize()

            # --- trajectory optimization ---
            insert_dv, rdvz_dv, st, et, er = trajectory_optimizer(
                origin_rot,
                ISO,
                detect_time,
                detect_time + max_time,
                **weights
            )

            insert_dv += plane_dv
        except:
            invalid_count += 1 # to be handled later
            continue

        stats = {
            "insertion dV": insert_dv,
            "rendezvous dV": rdvz_dv,
            "total dV": insert_dv + rdvz_dv,
            "launch time": st,
            "intercept time":et,
            "intercept distance":er 
        }
        ISO_stats.append(stats)


    # === plot data ===

    idVs = [x["insertion dV"] for x in ISO_stats]
    rdVs = [x["rendezvous dV"] for x in ISO_stats]


    plt.hist(idVs,bins=range(m.floor(min(idVs)), m.ceil(max(idVs))+1, 3),density=True)
    plt.xlabel("Insertion dV (km/s)")
    plt.ylabel("density")
    plt.show()

    plt.hist(rdVs,bins=range(m.floor(min(rdVs)), m.ceil(max(rdVs))+1, 3),density=True)
    plt.xlabel("relative velocity at intercept (km/s)")
    plt.ylabel("density")
    plt.show()
