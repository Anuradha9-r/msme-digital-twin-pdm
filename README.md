# Digital Twin-Driven Predictive Analytics for Coimbatore MSMEs

Code and data for the paper:

> R. Anuradha, A. Grace Selvarani, K. Vidya Kala, K. Nithya Kala, S. Lara Priyadharshini, and Ramesh Subramani,
> "Digital Twin-Driven Predictive Analytics for Production Improvement in Coimbatore MSMEs: A Physics-Informed
> Framework and Line-Level Simulation Study," *Engineering, Technology & Applied Science Research* (under review).

The repository contains everything needed to reproduce every table and figure in the paper:

1. **Data generation:** a synthetic machining process dataset (10,000 records) generated with the published
   AI4I 2020 procedure.
2. **Predictive models:** Logistic Regression, Random Forest, and Gradient Boosting failure classifiers, trained
   on raw sensor features and on physics-informed features derived from the digital twin state.
3. **Line digital twin:** a discrete-event simulation (SimPy) of a four-station pump-component machining line,
   comparing reactive, preventive, predictive, and twin-optimized maintenance.
4. **Figures:** scripts that redraw Figs. 1-4.

All random seeds are fixed, so the outputs are identical to those reported in the paper.

## Repository structure

```
msme-digital-twin-pdm/
├── src/
│   ├── generate_data.py   # synthetic process dataset (AI4I 2020 procedure)
│   ├── train_models.py    # classifiers, nested CV, permutation importance
│   ├── line_twin.py       # discrete-event line twin and maintenance policies
│   └── make_figures.py    # Figs. 1-4
├── data/
│   └── process_data.csv   # dataset used in the paper (10,000 records)
├── results/
│   ├── pm_results.json    # classifier results (Table III, Fig. 2)
│   └── twin_results.json  # line twin results (Table IV, Figs. 3-4)
├── figures/               # Figs. 1-4 as PNG
├── run_all.sh             # runs the full pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/<your-username>/msme-digital-twin-pdm.git
cd msme-digital-twin-pdm
pip install -r requirements.txt

bash run_all.sh            # full pipeline, about 15-25 minutes
```

Or run the steps individually:

```bash
python src/generate_data.py     # data/process_data.csv           (seconds)
python src/train_models.py      # results/pm_results.json         (2-3 min)
python src/line_twin.py         # results/twin_results.json       (10-20 min)
python src/make_figures.py      # figures/*.png                    (seconds)
```

To generate a different realisation of the dataset:

```bash
python src/generate_data.py --n 20000 --seed 1 --out data/my_data.csv
```

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Table I (dataset summary) | `generate_data.py` | `data/process_data.csv` |
| Table III, Fig. 2 (classifier performance, importance) | `train_models.py` | `results/pm_results.json` |
| Per-mode recall, accuracy comparison (Sections V-A, V-D) | `train_models.py` | `results/pm_results.json` |
| Table IV, Fig. 3 (line performance per policy) | `line_twin.py` | `results/twin_results.json` |
| Buffer search (Section V-B) | `line_twin.py` | `results/twin_results.json` |
| Fig. 4 (sensitivity analysis) | `line_twin.py` | `results/twin_results.json` |
| Figs. 1-4 | `make_figures.py` | `figures/` |

## Dataset

`data/process_data.csv` follows the generation procedure of the AI4I 2020 Predictive Maintenance Dataset
(S. Matzka, 2020, UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C).

| Column | Description |
|---|---|
| `type` | Product quality variant: L (50%), M (30%), H (20%) |
| `air_K` | Air temperature [K], random walk around 300 K (SD 2 K) |
| `process_K` | Process temperature [K], air temperature + 10 K + random walk (SD 1 K) |
| `speed_rpm` | Rotational speed [rpm], inversely coupled to torque, with noise |
| `torque_Nm` | Torque [Nm], normal(40, 10), no negative values |
| `tool_wear_min` | Tool wear [min], +2/3/5 min per process for L/M/H; tool replaced at 200-240 min |
| `TWF` | Tool wear failure: the tool fails at replacement time (probability 0.30) |
| `HDF` | Heat dissipation failure: (process - air temperature) < 8.6 K and speed < 1380 rpm |
| `PWF` | Power failure: torque x angular speed < 3500 W or > 9000 W |
| `OSF` | Overstrain failure: tool wear x torque > 11,000 / 12,000 / 13,000 minNm for L / M / H |
| `RNF` | Random failure, probability 0.001 |
| `failure` | 1 if any failure mode is active |

The dataset contains 467 failures (4.7%): TWF 38, HDF 186, PWF 123, OSF 120, RNF 11. A record may show more than
one mode.

## Notes

- The process data are synthetic, and the line twin parameters (Table II of the paper) are representative values
  for a small Coimbatore pump-component unit rather than measurements from a specific firm. Costs are indicative
  values in Indian rupees. Recalibrate them before applying the model to a real plant.
- Results were produced with the library versions in `requirements.txt`. Other versions of scikit-learn may give
  slightly different classifier figures.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{anuradha2026digitaltwin,
  author  = {Anuradha, R. and Grace Selvarani, A. and Vidya Kala, K. and Nithya Kala, K. and
             Lara Priyadharshini, S. and Subramani, Ramesh},
  title   = {Digital Twin-Driven Predictive Analytics for Production Improvement in Coimbatore MSMEs:
             A Physics-Informed Framework and Line-Level Simulation Study},
  journal = {Engineering, Technology \& Applied Science Research},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code and data are released under the MIT License (see `LICENSE`).

## Contact

Dr. R. Anuradha, Department of M.Tech Computer Science and Engineering, Sri Ramakrishna Engineering College,
Coimbatore, India. anuradha.r@srec.ac.in
