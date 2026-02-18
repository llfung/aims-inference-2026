import numpy as np
import pandas as pd


# Define constants
PI = np.pi


def wrap_phi(phi):
    """Map angles to [-pi, pi)."""
    return (phi + PI) % (2 * PI) - PI


# Function to simulate events
def run_simulation(
    n_events,
    mass=600.0,
    spin=1,
    rng=None,
):
    """
    Toy simulator for X' -> l + nu' events.

    Parameters
    ----------
    n_events : int
        Number of events to generate
    mass : float
        Mass of X' in GeV (400--1000)
    spin : int
        Spin hypothesis: 0 or 1
    rng : np.random.Generator or None
        Random number generator

    Returns
    -------
    data : ndarray, shape (n_events, 5)
        Columns:
        [muon_pt, muon_eta, muon_phi, MET, MET_phi]
    """
    if rng is None:
        rng = np.random.default_rng()

    # -------------------------------
    # Momentum scale from mass
    # -------------------------------
    scale = mass / 2.5  # arbitrary but smooth scaling

    # muon pT
    muon_pt = rng.gamma(
        shape=2.0,
        scale=scale / 2.0,
        size=n_events,
    )

        # -------------------------------
    # MET: partial correlation with muon pT
    # and mass-dependent shape
    # -------------------------------
    mass_min, mass_max = 400.0, 1000.0

    # Correlated component weight (decreases with mass)
    alpha = 0.8 - 0.3 * (mass - mass_min) / (mass_max - mass_min)
    alpha = np.clip(alpha, 0.4, 0.8)

    # Independent MET scale (increases with mass)
    beta = 0.15 * mass

    # Correlated recoil-like term
    met_corr = alpha * muon_pt * rng.normal(
        loc=1.0,
        scale=0.20,
        size=n_events,
    )

    # Independent invisible term
    met_indep = rng.gamma(
        shape=2.0,
        scale=beta / 4.0,
        size=n_events,
    )

    met = met_corr + met_indep
    met = np.clip(met, 0.0, None)

    # -------------------------------
    # muon angular variables
    # -------------------------------
    muon_phi = rng.uniform(-PI, PI, size=n_events)

    mass_min, mass_max = 400.0, 1000.0
    eta_width_min, eta_width_max = 1.0, 2.0

    # Linear interpolation
    eta_width = (
        eta_width_min
        + (eta_width_max - eta_width_min)
        * (mass - mass_min)
        / (mass_max - mass_min)
    )

    muon_eta = rng.normal(
        loc=0.0,
        scale=eta_width,
        size=n_events,
    )

    # -------------------------------
    # MET phi: spin-dependent structure
    # -------------------------------
    if spin == 0:
        # Spin-0: isotropic decay
        delta_phi = rng.uniform(-PI, PI, size=n_events)

    elif spin == 1:
        # Spin-1: correlated decay
        # Peak near back-to-back (Δφ ~ π)
        delta_phi = rng.vonmises(
            mu=PI,
            kappa=2.5,  # controls strength of correlation
            size=n_events,
        )
    else:
        raise ValueError("spin must be 0 or 1")

    met_phi = wrap_phi(muon_phi + delta_phi)

    # -------------------------------
    # Small detector-like smearing
    # -------------------------------
    muon_pt *= rng.normal(1.0, 0.05, size=n_events)
    met *= rng.normal(1.0, 0.08, size=n_events)

    # -------------------------------
    # Assemble output
    # -------------------------------
    # Provide output as pandas Dataframe
    data = pd.DataFrame(
        {
            "muon_pt": muon_pt,
            "muon_eta": muon_eta,
            "muon_phi": muon_phi,
            "MET": met,
            "MET_phi": met_phi,
        }
    )

    return data