import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u

def fetch_gaia_velocities(p_min=20, metallicity_range=(-1000, 1000), 
                          teff_range=(0, 1000000), limit=None):
    """
    Fetches Gaia DR3 kinematics and returns U, V, W in m/s.
    Tries Astroquery first, with a hard fallback to the AIP TAP mirror.
    """

    where_clause = f"""
    WHERE 
        parallax > {p_min}
        AND radial_velocity IS NOT NULL
        AND teff_gspphot BETWEEN {teff_range[0]} AND {teff_range[1]}
        AND mh_gspphot BETWEEN {metallicity_range[0]} AND {metallicity_range[1]}
        AND (parallax / parallax_error) > 10 
        AND visibility_periods_used > 8
        AND astrometric_n_obs_al > 100
        AND ruwe < 1.4
    """
    
    cols = "ra, dec, parallax, pmra, pmdec, radial_velocity"
    query_limit = f"TOP {limit} " if limit else ""
    query = f"SELECT {query_limit}{cols} FROM gaiadr3.gaia_source {where_clause}"
    
    results = None
    
    # Attempt Primary: Astroquery
    try:
        from astroquery.gaia import Gaia
        print("Fetching data via Astroquery...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            job = Gaia.launch_job_async(query)
            results = job.get_results()
    except Exception as e:
        print(f"Astroquery failed ({e}). Falling back to AIP mirror...")
        
    # Attempt Fallback: AIP TAP Mirror via PyVO
    if results is None:
        try:
            import pyvo
            tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")
            job = tap_service.submit_job(query)
            job.run()
            results = job.fetch_result().to_table()
        except Exception as e:
            raise RuntimeError(f"Data fetch failed entirely. Both Astroquery and AIP mirror are down: {e}")

    print(f"Successfully fetched {len(results)} stars.")

    # Convert coordinates to Galactic frame
    c = SkyCoord(
        ra=results['ra'], dec=results['dec'], 
        distance=(1000 / results['parallax']) * u.pc, 
        pm_ra_cosdec=results['pmra'], pm_dec=results['pmdec'], 
        radial_velocity=results['radial_velocity'], 
        frame='icrs'
    )

    gal = c.galactic
    
    # Extract U, V, W components in m/s
    U = gal.velocity.d_x.to(u.m/u.s).value
    V = gal.velocity.d_y.to(u.m/u.s).value
    W = gal.velocity.d_z.to(u.m/u.s).value
    
    return U, V, W