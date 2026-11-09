"""
Discrete-event digital twin of a representative Coimbatore pump-component machining
line (Sections III-D, IV-C, V-B and V-C; Tables II and IV; Figs. 3 and 4).

Line: CNC turning -> VMC milling -> drilling/tapping -> inspection, with finite
buffers between stations. Machining stations fail with Weibull(eta, beta = 2)
time-to-failure in operating hours; repairs are log-normal (technician and
spare-part delay). Parts machined within 10 h before a failure have a higher
defect probability. Products run in batches of 60 with a setup per batch.

Maintenance policies compared:
  S0  reactive         run to failure, then repair
  S1  preventive       time-based renewal after 0.6 * eta operating hours
  S2  predictive       classifier detects an impending failure with probability
                       = recall, 2-8 h ahead; false alarms at the rate implied by
                       precision, each costing a short inspection stop
  S3  twin-optimized   S2 + alarm-driven maintenance merged into a changeover
                       within 10 h, + buffer sizes chosen by a what-if search

Each policy: six months (2,496 planned hours, 40 h warm-up), 30 replications with
seeds 0..29; results are means with 95 % confidence intervals.

Usage:
    python src/line_twin.py            # full study, writes results/twin_results.json
Runtime: roughly 10-20 minutes on a laptop (the 125-combination buffer search
and the sensitivity grids dominate).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import simpy

ROOT = Path(__file__).resolve().parents[1]

H = 26 * 6 * 16.0        # 6 months, 26 days/month, 2 shifts x 8 h -> planned hours
WARM = 40.0
STATIONS = [  # name, (tri min, mode, max) minutes, weibull eta (op-h), fails?
    ("CNC turning",      (5.8, 6.5, 7.6), 140.0, True),
    ("VMC milling",      (6.8, 7.5, 8.8), 110.0, True),
    ("Drilling/tapping", (4.0, 4.5, 5.4), 180.0, True),
    ("Inspection",       (2.6, 3.0, 3.6), None,  False),
]
BETA = 2.0
# Line twin parameters (Table II); costs in Indian rupees
P = dict(repair_mean=6.0, repair_cv=0.5,      # h, breakdown incl. technician & spares wait
         pm_dur=1.5, fa_dur=0.75,             # h, planned maintenance / false-alarm inspection
         lead=(2.0, 8.0),                     # h, alarm lead time before failure
         degr_win=10.0,                       # h before failure with elevated defects
         q_base=0.015, q_degr=0.06,
         margin=450.0, c_break=12000.0, c_pm=2500.0, c_fa=1000.0, c_scrap=800.0,
         pm_frac=0.6, batch=60, setup=(0.4, 0.6, 1.0), opp_window=10.0, ideal_bn=6.8)
BUF0 = [5, 5, 5]

def lognorm(rng, mean, cv):
    s2 = np.log(1 + cv ** 2); return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2))

def run(policy, recall=0.889, precision=0.998, bufs=BUF0, seed=0, params=P):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    stores = [simpy.Store(env, capacity=b) for b in bufs]
    k = dict(out=0, good=0, scrap=0, fail=0, pm=0, fa=0, down=np.zeros(4), unpl=np.zeros(4),
             busy=np.zeros(4), wipt=0.0, setup_h=0.0)
    def rec(key, v=1, i=None):
        if env.now >= WARM:
            if i is None: k[key] += v
            else: k[key][i] += v

    def machine(i):
        name, tri, eta, fails = STATIONS[i]
        def new_life():
            ttf = eta * rng.weibull(BETA) if fails else np.inf
            st = dict(op=0.0, ttf=ttf, alarm=np.inf, next_fa=np.inf)
            if fails and policy in ("S2", "S3"):
                if rng.random() < recall:
                    st["alarm"] = max(0.0, ttf - rng.uniform(*params["lead"]))
                fa_rate = recall * (1 - precision) / max(precision, 1e-6) / (eta * 0.886)
                st["next_fa"] = rng.exponential(1 / fa_rate) if fa_rate > 0 else np.inf
            if fails and policy == "S1":
                st["pm_at"] = params["pm_frac"] * eta
            return st
        st = new_life(); n_done = 0
        while True:
            if i == 0:
                part = dict(t0=env.now, bad=False)
            else:
                part = yield stores[i - 1].get()
            pt = rng.triangular(*tri) / 60.0
            if n_done > 0 and n_done % params["batch"] == 0 and i < 3:
                su = rng.triangular(*params["setup"])
                if (policy == "S3" and fails and st["op"] + params["opp_window"] >= st["alarm"]):
                    # twin schedules the alarm-driven PM inside the changeover window
                    d = max(su, params["pm_dur"]); yield env.timeout(d)
                    rec("pm"); rec("down", d, i); rec("setup_h", su); st = new_life()
                else:
                    yield env.timeout(su); rec("down", su, i); rec("setup_h", su)
            n_done += 1
            # planned maintenance triggers (checked before starting the part)
            if policy == "S1" and fails and st["op"] >= st["pm_at"]:
                yield env.timeout(params["pm_dur"]); rec("pm"); rec("down", params["pm_dur"], i); st = new_life()
            if policy in ("S2", "S3") and fails:
                if st["op"] >= st["alarm"]:
                    yield env.timeout(params["pm_dur"]); rec("pm"); rec("down", params["pm_dur"], i); st = new_life()
                elif st["op"] >= st["next_fa"]:
                    yield env.timeout(params["fa_dur"]); rec("fa"); rec("down", params["fa_dur"], i)
                    st["next_fa"] = st["op"] + rng.exponential(1 / (recall * (1 - precision) / max(precision, 1e-6) / (eta * 0.886)))
            if fails and st["op"] + pt >= st["ttf"]:
                run_t = max(0.0, st["ttf"] - st["op"])
                yield env.timeout(run_t); rec("busy", run_t, i)
                rep = lognorm(rng, params["repair_mean"], params["repair_cv"])
                yield env.timeout(rep)
                rec("fail"); rec("down", rep, i); rec("unpl", rep, i); rec("scrap")
                st = new_life()
                continue          # part lost
            yield env.timeout(pt); rec("busy", pt, i); st["op"] += pt
            if fails and (st["ttf"] - st["op"]) < params["degr_win"]:
                if rng.random() < params["q_degr"] - params["q_base"]: part["bad"] = True
            if i < 3:
                yield stores[i].put(part)
            else:
                if rng.random() < params["q_base"]: part["bad"] = True
                rec("out")
                if part["bad"]: rec("scrap")
                else:
                    rec("good")
                    rec("wipt", env.now - part["t0"])
    for i in range(4): env.process(machine(i))
    env.run(until=H)
    T = H - WARM
    ideal_bn = params["ideal_bn"] / 60.0
    good = k["good"]
    avail_bn = 1 - k["down"][1] / T
    oee = good * ideal_bn / T
    cost = (k["fail"] * params["c_break"] + k["pm"] * params["c_pm"] + k["fa"] * params["c_fa"]
            + k["scrap"] * params["c_scrap"])
    scale = (H / T)  # report over full horizon
    months = 6
    return dict(good_pm=good * scale / months, out_pm=k["out"] * scale / months,
                oee=oee, avail_bn=avail_bn, perf_bn=(k["busy"][1] / max(T - k["down"][1], 1e-9)) * params["ideal_bn"] / 7.5,
                quality=good / max(k["out"], 1), fails_pm=k["fail"] * scale / months,
                pm_pm=k["pm"] * scale / months, fa_pm=k["fa"] * scale / months,
                unpl_h_pm=k["unpl"].sum() * scale / months, down_h_pm=k["down"].sum() * scale / months,
                maint_cost_pm=cost * scale / months,
                net_pm=(good * params["margin"] - cost) * scale / months,
                lead_h=k["wipt"] / max(good, 1), bufs=list(bufs))

def reps(policy, n=30, **kw):
    rs = [run(policy, seed=s, **kw) for s in range(n)]
    keys = [k for k in rs[0] if k != "bufs"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rs]); m = v.mean(); h = 2.045 * v.std(ddof=1) / np.sqrt(n)
        out[k] = [float(m), float(h)]
    return out

if __name__ == "__main__":
    res = {}
    for pol in ["S0", "S1", "S2"]:
        res[pol] = reps(pol); print(pol, {k: round(v[0], 3) for k, v in res[pol].items()})
    # S3: twin-based what-if search over the 125 buffer allocations; pick the smallest total
    # buffer capacity reaching >= 99.5 % of the best throughput
    grid = []
    for b in itertools.product([2, 5, 10, 15, 20], repeat=3):
        r = reps("S3", n=8, bufs=list(b))
        grid.append((list(b), r["good_pm"][0], r["lead_h"][0]))
    gmax = max(g[1] for g in grid)
    cand = [g for g in grid if g[1] >= 0.995 * gmax]
    bestb = min(cand, key=lambda g: (sum(g[0]), g[2]))[0]
    print("best buffers", bestb, gmax)
    res["S3"] = reps("S3", bufs=bestb); res["S3_bufs"] = bestb
    print("S3", {k: round(v[0], 3) for k, v in res["S3"].items() if isinstance(v, list)})
    res["buffer_grid"] = grid
    # sensitivity: recall x precision -> good output gain vs S0
    base = res["S0"]["good_pm"][0]
    sens = []
    for rc in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        row = []
        for pr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r = reps("S2", n=10, recall=rc, precision=pr)
            row.append(100 * (r["good_pm"][0] / base - 1))
        sens.append(row); print(rc, [round(x, 1) for x in row])
    res["sens"] = dict(recall=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], precision=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], gain=sens)
    # sensitivity to MSME repair delay
    rep_s = []
    for rm in [2, 4, 6, 8, 12]:
        pp = dict(P); pp["repair_mean"] = rm
        a = reps("S0", n=10, params=pp)["good_pm"][0]; b = reps("S2", n=10, params=pp)["good_pm"][0]
        rep_s.append([rm, a, b, 100 * (b / a - 1)]); print("repair", rm, round(100 * (b / a - 1), 1))
    res["repair_sens"] = rep_s
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "twin_results.json", "w"), indent=1)
    print("Wrote", ROOT / "results" / "twin_results.json")
