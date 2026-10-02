# Hybrid Synthetic-Covariate Augmentation for Doubly-Robust Off-Policy Evaluation of Lending Policies

MSc dissertation project, Queen Mary University of London.
Supervisor: David Mguni.

This project tests whether a **hybrid synthetic-covariate augmentation** method, previously shown to improve static average treatment effect estimation under low propensity overlap, also improves **doubly-robust off-policy evaluation (DR-OPE)** of lending policies. It uses LendingClub's historical accepted and rejected loan applications.

---

## Research question

> Does hybrid synthetic-covariate augmentation improve the accuracy and stability of doubly-robust off-policy evaluation of lending policies under low propensity overlap, and does it outperform both standard DR (no augmentation) and naive fully-generative augmentation, as it did in the static setting?

Historical approve/reject decisions are treated as a **behaviour policy** π_b. The value of alternative **target policies** π_e is estimated using only the logged data. The hybrid method:

1. generates **synthetic covariates only** (never outcomes) near real observations with extreme propensity;
2. trains the outcome model on the **full** dataset;
3. pairs each extreme-propensity observation with its nearest synthetic covariate point and uses the outcome model's predictions in the outcome-regression part of DR.

It is compared against:

| Approach | Role |
|---|---|
| Direct Method, IPS, standard DR | Baselines |
| Hybrid-augmented DR | Method under test |
| Naive fully-generative DR | Negative control |
| DR with shrinkage (Su et al., 2020) | Established non-generative fix |

---

## Project status

| Stage | Status |
|---|---|
| Data cleaning and accepted + rejected concatenation | ✅ Written; logic tested on sample data, full run pending |
| Propensity model (behaviour policy) | ⏳ Next |
| DM / IPS / DR estimators with unit tests | ⏳ Planned |
| Bayesian extension and calibration | ⏳ Planned |
| Low-propensity diagnostic | ⏳ Planned |
| Hybrid covariate generator + pairing | ⏳ Planned |
| Naive-generative and shrinkage baselines | ⏳ Planned |
| Comparative evaluation | ⏳ Planned |

---

## Data

**Source:** [LendingClub loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) on Kaggle (`wordsforthewise/lending-club`), covering 2007 – 2018 Q4.

| File | Rows | Columns |
|---|---|---|
| `accepted_2007_to_2018Q4.csv` | ~2.26M | 151 |
| `rejected_2007_to_2018Q4.csv` | ~27.6M | 9 |

The data is **not included in this repository**. It is downloaded automatically with `kagglehub` (see [Getting started](#getting-started)).

### Combined dataset (propensity model)

The rejected file only has 9 columns, so both files are reduced to a shared schema and concatenated with an `accepted` flag. All years (2007 – 2018) are kept, about 30M rows; `in_ope_period` = 1 marks applications before 2015, the period that matches the outcome dataset.

| Column | From rejected | From accepted |
|---|---|---|
| `amount` | Amount Requested | `loan_amnt` |
| `date`, `year`, `month` | Application Date | `issue_d` |
| `purpose` | Loan Title, mapped to a category | `title` (or `purpose` if empty), mapped the same way |
| `risk_score` | Risk_Score | midpoint of `fico_range_low` / `fico_range_high` |
| `dti` | Debt-To-Income Ratio (`"38.64%"` → 38.64) | `dti` |
| `zip3` | Zip Code (`"481xx"` → `481`) | `zip_code` |
| `state` | State | `addr_state` |
| `emp_length` | Employment Length (`< 1 year` → 0, `10+ years` → 10) | `emp_length` |
| `accepted` | 0 | 1 |
| `in_ope_period` | 1 if before 2015 | 1 if before 2015 |

### Outcome dataset (outcome model)

Accepted loans only, with the shared columns plus 13 **pre-decision** features: `term`, `annual_inc`, `home_ownership`, `verification_status`, `lc_purpose`, `revol_bal`, `revol_util`, `open_acc`, `total_acc`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `mort_acc`.

Target: `default` = 1 if the loan was Charged Off or Defaulted, 0 if Fully Paid.

---

## Data processing decisions

Each decision below is there to stop the models learning from artefacts of how the data was recorded, rather than from real applicant information.

**Leakage prevention**
- **`Policy Code` is dropped.** It is 0 for every rejected row and 1 for every accepted row, so it would reveal the label.
- **Dates are rounded to the month.** Accepted loans only record an issue month (always day 1), so an exact day would identify a rejected application.
- **`grade`, `sub_grade`, `int_rate` and `installment` are excluded.** They are LendingClub's own pricing decision, not information about the applicant before the decision.
- **Post-issuance columns are excluded**, such as payments, recoveries, hardship and settlement. They only exist because a loan was issued.

**Consistency between accepted and rejected**
- **The same title-to-purpose keyword mapper is applied to both files.** Rejected applications have no `purpose` field, and using LendingClub's official purpose only for accepted loans would make the two groups systematically different. The official purpose is kept separately as `lc_purpose` in the outcome dataset.

**Invalid and missing values**
- **Placeholders become NaN rather than being filled.** A DTI below 0 (LendingClub uses -1 for missing) or above 100, and risk scores outside 300 – 990 (0 is a placeholder), are treated as missing.
- **Gradient-boosted models handle missing values natively**, so no imputation is used.

**Year cut-off (unfinished loans)**
- **The data ends in December 2018, and loans still running then have no final outcome.** Those are mostly good loans, because bad loans fail early, so keeping only finished loans from recent years would inflate the default rate.
- **The outcome dataset keeps a loan only if it was issued at least its full term plus 12 months before December 2018.** That means 36-month loans issued before 2015 and 60-month loans issued before 2013.
- **The combined dataset keeps every year**, because acceptance is known on the day of the decision, and about 88% of rejected applications came in 2015 – 2018. `in_ope_period` marks the pre-2015 rows. The propensity model can train on all years, with `year` as a feature, but IPS/DR estimates and calibration checks use only the flagged rows, so both models describe the same applicants.
- **The settings are in a single cell** (`DATA_END`, `BUFFER_MONTHS`, `CUTOFF`).

### Known limitations

- **`Risk_Score` changes scale in November 2013.** From then on, rejected applications use a VantageScore, while accepted loans always use FICO. With all years kept, this affects most rejected rows. Missingness in rejected risk scores also rises sharply over time: about 67% overall, but about 10% before 2015.
- **The acceptance rule changed over time**, and most propensity training rows come from 2015 – 2018.
- **Missingness differs by group.** Risk scores are often missing for rejected applicants and almost never for accepted ones, so missingness itself can act as a signal.
- **Date meaning differs.** Rejected loans have an application date and accepted loans an issue (funding) date.
- **Classes are imbalanced.** There are roughly 12 rejected applications for every accepted loan.
- **Policy value estimates describe 2007 – 2014 borrowers**, because of the year cut-off on outcomes.
- **Outcomes are only observed for accepted borrowers.** This is the selection problem that IPS/DR is designed to address, and it only works where accepted and rejected applicants overlap.

---

## Getting started

### Google Colab (recommended)

The loading cells use compact data types to keep memory down, but the combined dataset is about 30 million rows, so a **high-RAM runtime** is safer.

1. Open `cleaning/data_processing.ipynb` in Colab.
2. Optional: set `OUT_DIR` in the settings cell to a Google Drive folder. The output files are too large to download reliably from the browser.
3. Run all cells. The dataset is downloaded with `kagglehub`, which may ask for Kaggle credentials.
4. Outputs are written to `OUT_DIR`:
   - `lending_club_combined.parquet`: accepted + rejected, shared schema
   - `lending_club_outcome.parquet`: accepted loans with outcomes

### Locally

```bash
git clone https://github.com/KP-365/hybrid-ope-lending.git
cd hybrid-ope-lending
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook cleaning/data_processing.ipynb
```

---

## Repository structure

```
hybrid-ope-lending/
├── cleaning/
│   └── data_processing.ipynb   # cleaning, concatenation, outcome dataset
├── requirements.txt
├── .gitignore
└── README.md
```

Planned as the project develops: `src/` (data, propensity, outcome, estimators, generators, evaluation), `configs/` (YAML experiment configs), `tests/` (unit tests against hand-computed examples), and `results/`.

---

## Key references

1. Xu (2026). *Generative Synthetic Data for Causal Inference: Pitfalls, Remedies, and Opportunities.* arXiv:2604.23904.
2. Xu, Skoularidou, Cuesta-Infante & Veeramachaneni (2019). *Modeling Tabular Data using Conditional GAN.* NeurIPS.
3. Su, Dimakopoulou, Krishnamurthy & Dudík (2020). *Doubly Robust Off-Policy Evaluation with Shrinkage.* ICML.
4. Dudík, Langford & Li (2011). *Doubly Robust Policy Evaluation and Learning.* ICML.
5. Bang & Robins (2005). *Doubly Robust Estimation in Missing Data and Causal Inference Models.* Biometrics, 61(4).
6. Banasik, Crook & Thomas (2003). *Sample Selection Bias in Credit Scoring Models.* Journal of the Operational Research Society, 54(8).
7. Siddiqi (2006). *Credit Risk Scorecards.* Wiley.

---

## Data licence

LendingClub data is subject to the terms of its Kaggle source and is not redistributed here.
