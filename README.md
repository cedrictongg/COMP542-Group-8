# COMP542-Group-8: NBA Hall of Fame Prediction

A machine learning project for CSUN's COMP 542 (Machine Learning) course that predicts NBA player Hall of Fame induction from historical career statistics, awards, and accolades using Basketball-Reference-style player data.

## Overview

This project builds a binary classification pipeline that predicts whether an NBA player will be inducted into the Hall of Fame (`hof`), based on aggregated career stats (points, rebounds, assists, advanced metrics like Win Shares and VORP), All-Star selections, All-NBA/All-Defense/All-Rookie team appearances, and MVP/award voting shares. The workflow spans data merging, feature engineering, feature selection via permutation importance, outlier removal, training and tuning six classifiers (Logistic Regression for feature selection, SVC, KNN, Random Forest, HistGradientBoosting, XGBoost, and LightGBM), threshold optimization, and final model evaluation.

## Repository Structure

```
COMP542-Group-8/
├── data/                                  # Raw Basketball-Reference-style CSVs
│   ├── Advanced.csv
│   ├── All-Star Selections.csv
│   ├── End of Season Teams.csv
│   ├── End of Season Teams (Voting).csv
│   ├── Opponent Totals.csv
│   ├── Per 100 Poss.csv
│   ├── Per 36 Minutes.csv
│   ├── Player Award Shares.csv
│   ├── Player Career Info.csv
│   ├── Player Per Game.csv
│   ├── Player Play By Play.csv
│   ├── Player Season Info.csv
│   ├── Player Shooting.csv
│   ├── Player Totals.csv
│   ├── Team Abbrev.csv
│   └── Team Totals.csv
├── docs/                                  # Course/project administration documents
│   ├── 1 GROUP SignUp.docx
│   ├── 3 Project Topic Form.docx
│   └── Group 8 Sign Up.docx
├── NBA Prediction PreCompute.ipynb        # Data merging & feature engineering (builds NBA_data)
├── NBA Prediction.ipynb                   # Feature selection, model training, tuning, evaluation
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

## Data

The `data/` directory contains player- and team-level season and career statistics scraped from Basketball-Reference, spanning career info (height, weight, position, career span), season-by-season totals, advanced stats (Win Shares, VORP, BPM, PER, usage %), All-Star selections, End-of-Season awards (All-NBA/All-Defense/All-Rookie teams), and MVP/other award voting shares.

`NBA Prediction.ipynb` merges these on `player_id` to build a single player-level dataset (`NBA_data`, 5,416 rows × 58 columns) with:

- **Target**: `hof` — Hall of Fame status (boolean, cast to int).
- **Career totals**: games, minutes, points, rebounds, assists, steals, blocks, turnovers, shooting splits (`career_g`, `career_pts`, `career_trb`, `career_fg_pct`, etc.), computed by summing per-season totals grouped by `player_id`.
- **Advanced metrics**: summed stats (`career_ows`, `career_dws`, `career_ws`, `career_vorp`) and minutes-weighted average efficiency stats (`career_avg_per`, `career_avg_ts_percent`, `career_avg_bpm`, `career_avg_usg_percent`, etc.), plus `peak_season_ws` (best single-season Win Shares).
- **Accolades**: `n_seasons`, `n_all_star`, `n_all_nba`, `n_all_defense`, `n_all_rookie`, `n_first_team_all_nba`, `max_award_share`, `total_award_share`, `n_award_wins`, `mvp_max_share`, `n_mvp_wins`.

Missing accolade/award values (players with no All-Star appearances or awards) are filled with 0, and players with no recorded career stats (`career_g`) are dropped.

## Notebooks

| Notebook | Purpose |
|---|---|
| `NBA Prediction PreCompute.ipynb` | Loads raw CSVs from `data/` (forcing the working directory to the notebook's own folder), computes career length, sums career totals and advanced stats, computes minutes-weighted efficiency averages and peak-season Win Shares, counts seasons/All-Star/All-NBA/All-Defense/All-Rookie/First-Team appearances, sums award shares and MVP shares, merges everything into `NBA_data`, and fills missing values. |
| `NBA Prediction.ipynb` | Core modeling notebook. Splits `NBA_data` into train/test (80/20, stratified on `hof`), fits a `StandardScaler` + `LogisticRegression(class_weight="balanced")` pipeline, ranks features via `permutation_importance` (scored on ROC-AUC), hand-selects 15 `important_selected_features` (e.g., `career_vorp`, `n_seasons`, `career_pf`, `mvp_max_share`, `n_all_nba`, `n_all_star`, `peak_season_ws`), restricts the dataset to HOF-eligible players (`to <= 2018`), removes outliers via `LocalOutlierFactor`, trains and tunes six models (SVC, KNN, Random Forest, HistGradientBoostingClassifier, XGBoost, LightGBM) with `GridSearchCV`/`RandomizedSearchCV`, calibrates HistGradientBoosting with `CalibratedClassifierCV`, tunes decision thresholds via precision/recall/F1 curves, selects the best model by cross-validated ROC-AUC, and generates Hall-of-Fame probability predictions for individual players (e.g., Michael Jordan, Bill Walton) and league-wide leaderboards of non-Hall-of-Famers most likely to be inducted. |

## Methodology

1. **Data integration** — Merge multiple Basketball-Reference CSVs on `player_id` to build the 58-column `NBA_data` career-level feature table.
2. **Baseline feature ranking** — A `StandardScaler` + `LogisticRegression(class_weight="balanced")` pipeline is fit on an 80/20 stratified split, then `permutation_importance` (`n_repeats=20`, scored on `roc_auc`) ranks features; the top signals include `career_vorp`, `n_seasons`, `career_pf`, `career_fta`, `career_x3pa`, `mvp_max_share`, `career_dws`, and `career_ast_per_g`.
3. **Feature selection** — 15 features are hand-picked from the importance ranking (`important_selected_features`) for use in the tree-based/kernel models.
4. **Eligibility filter & outlier removal** — The dataset is restricted to players whose careers ended by 2018 (HOF-eligible), then `LocalOutlierFactor` (`n_neighbors=20`) removes anomalous rows from the training and tuning splits (84 and 75 outliers removed, respectively) before fitting SVC/KNN/Random Forest.
5. **Class imbalance handling** — XGBoost uses `scale_pos_weight` (ratio of non-HOF to HOF players in training data); LightGBM uses `is_unbalance=True`; HistGradientBoosting uses `sample_weight` from `compute_sample_weight(class_weight="balanced")`.
6. **Hyperparameter tuning** — `GridSearchCV` tunes SVC, KNN, Random Forest, and HistGradientBoosting; `RandomizedSearchCV` (40 iterations, 5-fold, scored on `roc_auc`) tunes XGBoost and LightGBM.
7. **Calibration & thresholding** — HistGradientBoosting predictions are calibrated with `CalibratedClassifierCV` (sigmoid method); for the other models, a threshold sweep (0.0–0.9) on validation-set predicted probabilities selects the threshold maximizing F1 score via `FixedThresholdClassifier`.
8. **Model selection** — All six tuned models are compared via 5-fold cross-validated ROC-AUC on the training set; Random Forest was selected as the best model in the current run (test accuracy ≈ 0.984).
9. **Inference** — The selected model produces Hall-of-Fame probabilities for named players (e.g., Michael Jordan ≈ 0.989, Bill Walton ≈ 0.38) and ranks currently-active or recently-retired non-Hall-of-Famers by predicted HOF probability (e.g., LeBron James, James Harden, Russell Westbrook rank highest).

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/cedrictongg/COMP542-Group-8.git
cd COMP542-Group-8
pip install -r requirements.txt
```

Then launch Jupyter to run the notebooks in order:

```bash
jupyter notebook
```

Run `NBA Prediction PreCompute.ipynb` first to build the merged `NBA_data` dataset, then `NBA Prediction.ipynb` for feature selection, model training, tuning, and evaluation.

## Hardware Specifications

### Course Minimum Requirements

| Component | Minimum Requirement |
|---|---|
| CPU | 13th Gen Intel or AMD Ryzen 7000 series or higher |
| RAM | 16 GB or higher |
| Storage | 1 TB |
| GPU | NVIDIA GPU with 6 GB VRAM or higher |

### Team Hardware Configurations

The project was developed and trained across the following team member machines, each of which meets or exceeds the course's minimum requirements:

| Machine | CPU | GPU | RAM | Storage |
|---|---|---|---|---|
| 1 | Intel Core i7-13700HX | NVIDIA GeForce RTX 4070 Laptop GPU | 32 GB DDR5 | 5 TB M.2 |
| 2 | Apple M1 chip (8-core: 4 performance + 4 efficiency cores) | 8-core Apple integrated GPU | 8 GB LPDDR4X | 256 GB PCIe-based SSD |
| 3 | AMD Ryzen 7 9700X | NVIDIA GeForce RTX 5070 Ti | 64 GB DDR5 | 4 TB M.2 |

## Results

Six models were tuned and compared via 5-fold cross-validated ROC-AUC on the training set:

| Model | Cross-Validated ROC-AUC |
|---|---|
| Random Forest | 0.976 |
| HistGradientBoosting | 0.970 |
| XGBoost | 0.967 |
| LightGBM | 0.966 |
| Radial SVC | 0.956 |
| KNN | 0.955 |

## Course Context

This repository was developed for **COMP 542: Machine Learning** at California State University, Northridge (CSUN). The `docs/` folder contains group sign-up and project topic administrative forms submitted as part of the course requirements.
