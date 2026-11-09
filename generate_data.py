"""
Generate the synthetic machining process dataset used in the paper

    "Digital Twin-Driven Predictive Analytics for Production Improvement in
     Coimbatore MSMEs: A Physics-Informed Framework and Line-Level Simulation Study"

The generation procedure follows the published description of the AI4I 2020
Predictive Maintenance Dataset (S. Matzka, 2020; UCI Machine Learning Repository,
https://doi.org/10.24432/C5HS5C):

  * Product quality type L / M / H in a 50 / 30 / 20 % mix.
  * Air temperature [K]: random walk normalised to SD 2 K around 300 K.
  * Process temperature [K]: air temperature + 10 K + random walk normalised to SD 1 K.
  * Torque [Nm]: normal(40, 10), no negative values.
  * Rotational speed [rpm]: derived from a near-constant power, i.e. inversely
    coupled to torque, plus normal noise.
  * Tool wear [min]: accumulates by 2 / 3 / 5 min per process for L / M / H;
    the tool is replaced at a random wear level between 200 and 240 min.

Five failure modes are applied with the published rules:

  TWF  tool wear failure      : at tool replacement, the tool fails instead of
                                being replaced (probability 0.30)
  HDF  heat dissipation       : (process - air temperature) < 8.6 K AND speed < 1380 rpm
  PWF  power failure          : power = torque * omega < 3500 W or > 9000 W
  OSF  overstrain failure     : tool wear * torque > 11000 / 12000 / 13000 minNm for L / M / H
  RNF  random failure         : probability 0.001 per process

The machine-failure label is 1 if any of the five modes is active.

Usage
-----
    python src/generate_data.py                       # reproduces the paper's dataset
    python src/generate_data.py --n 20000 --seed 1    # a different realisation

With the default arguments (n = 10000, seed = 2026) the output is identical to
data/process_data.csv used for every result in the paper.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# Parameters of the generation procedure
# ----------------------------------------------------------------------------
TYPE_MIX = {"L": 0.5, "M": 0.3, "H": 0.2}      # product quality mix
WEAR_INCREMENT = {"L": 2, "M": 3, "H": 5}        # tool wear added per process [min]
TOOL_LIFE_RANGE = (200.0, 240.0)                 # tool replaced at this wear [min]
P_TOOL_FAILS = 0.30                              # tool fails instead of being replaced
HDF_DT_LIMIT, HDF_SPEED_LIMIT = 8.6, 1380.0      # heat dissipation thresholds [K], [rpm]
PWF_LOW, PWF_HIGH = 3500.0, 9000.0               # power failure limits [W]
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000} # overstrain limits [min*Nm]
P_RANDOM_FAILURE = 0.001


def random_walk(n, sd_target, rng):
    """Random walk rescaled to zero mean and the target standard deviation."""
    w = np.cumsum(rng.normal(0, 1, n))
    w = (w - w.mean()) / w.std()
    return w * sd_target


def generate(n=10000, seed=2026):
    """Return a DataFrame with process variables, failure modes and the label."""
    rng = np.random.default_rng(seed)

    # --- process variables -------------------------------------------------
    ptype = rng.choice(list(TYPE_MIX), size=n, p=list(TYPE_MIX.values()))
    air = 300 + random_walk(n, 2.0, rng)
    proc = air + 10 + random_walk(n, 1.0, rng)
    torque = np.clip(rng.normal(40, 10, n), 3.8, None)
    # speed from (approximately constant) power -> inverse coupling with torque
    speed = 1540 - 16.5 * (torque - 40) + rng.normal(0, 95, n)
    speed = np.clip(speed, 1160, 2890)

    # --- tool wear and tool wear failure (TWF) ------------------------------
    wear = np.zeros(n)
    twf = np.zeros(n, int)
    w = 0.0
    threshold = rng.uniform(*TOOL_LIFE_RANGE)
    for i in range(n):
        w += WEAR_INCREMENT[ptype[i]]
        if w >= threshold:                      # tool reaches end of life
            if rng.random() < P_TOOL_FAILS:     # ... and fails instead of being replaced
                twf[i] = 1
            wear[i] = w
            w = 0.0                             # new tool
            threshold = rng.uniform(*TOOL_LIFE_RANGE)
        else:
            wear[i] = w

    # --- remaining failure modes --------------------------------------------
    dT = proc - air
    omega = speed * 2 * np.pi / 60
    power = torque * omega
    strain = wear * torque
    osf_lim = np.select([ptype == "L", ptype == "M"], [OSF_LIMIT["L"], OSF_LIMIT["M"]], OSF_LIMIT["H"])

    hdf = ((dT < HDF_DT_LIMIT) & (speed < HDF_SPEED_LIMIT)).astype(int)
    pwf = ((power < PWF_LOW) | (power > PWF_HIGH)).astype(int)
    osf = (strain > osf_lim).astype(int)
    rnf = (rng.random(n) < P_RANDOM_FAILURE).astype(int)
    failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

    return pd.DataFrame(dict(
        type=ptype, air_K=air, process_K=proc, speed_rpm=speed, torque_Nm=torque,
        tool_wear_min=wear, TWF=twf, HDF=hdf, PWF=pwf, OSF=osf, RNF=rnf, failure=failure))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10000, help="number of records (default 10000)")
    ap.add_argument("--seed", type=int, default=2026, help="random seed (default 2026, as in the paper)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "process_data.csv", help="output CSV path")
    a = ap.parse_args()

    df = generate(a.n, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    modes = {m: int(df[m].sum()) for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]}
    print(f"Wrote {len(df)} records to {a.out}")
    print(f"Failures: {int(df.failure.sum())} ({100 * df.failure.mean():.1f}%)  modes: {modes}")


if __name__ == "__main__":
    main()
