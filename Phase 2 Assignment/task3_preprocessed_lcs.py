# =============================================================================
# TASK 6 (MIDDLE ROW): ORIGINAL (UNMODIFIED) eLCS ON THE PREPROCESSED DATASET
# -----------------------------------------------------------------------------
# The Task 6 rubric requires THREE separate LCS rows, not two:
#   1. Original LCS on the RAW dataset               -> task2_raw_lcs.py
#   2. Original LCS on the PREPROCESSED dataset       -> THIS FILE
#   3. Improved LCS on the preprocessed/feature-engineered dataset -> task4_improved_lcs.py
#
# This file is row 2. It uses the exact same, UNMODIFIED eLCS algorithm and
# the exact same settings as task2_raw_lcs.py (same N, same iterations, same
# seeds) -- the ONLY thing that changes here is that the data has gone
# through real Task 3 preprocessing first. This isolates the effect of
# "cleaning the data" on its own, separate from the effect of "selecting a
# small, curated feature set" (which is what task4_improved_lcs.py tests).
#
# WHAT COUNTS AS "PREPROCESSED" HERE, AND WHY (see Section 4 below for the
# actual code):
#   - Checked the full raw dataset for missing values and duplicate rows.
#     There are none -- this is reported explicitly rather than skipped,
#     since the Task 3 checklist asks preprocessing to "address" these, and
#     confirming there's nothing to fix IS a data-quality finding worth
#     stating.
#   - Outlier handling: the 5 core numeric predictors are winsorized (capped
#     at their IQR bounds) rather than left as extreme raw values or removed
#     outright. Capping keeps every row in the dataset (no lost sample size)
#     while reducing the influence of a small number of extreme values on
#     eLCS's rule boundaries.
#   - Leakage handling: 10 "outcome-family" columns are dropped
#     (revenue_growth_percent, cost_reduction_percent, customer_satisfaction,
#     time_saved_per_week, employee_satisfaction_score, jobs_created,
#     jobs_displaced, reskilled_employees, remote_work_percentage,
#     innovation_score). These all measure a similar kind of "AI paid off"
#     outcome as productivity_change_percent itself, so keeping them in would
#     let the model lean on other outcome measures rather than genuine
#     adoption-side predictors -- a leakage risk the Task 3 checklist
#     explicitly asks us to guard against.
#   - This row deliberately does NOT reduce down to the curated 7-column set
#     -- it keeps the broad ~30-column feature space (just cleaned), so the
#     comparison against Task 4's 7-column "improved" run isolates the effect
#     of feature *selection* specifically, on top of this row's cleaning.
#
# NOTE FOR THE TEAM: every line below has a comment explaining what it does.
# Read through this and rewrite the comments in your own words before you
# submit -- the lecturer wants to see that you understand the code, not just
# that it runs.
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import pandas as pd                                        # pandas: loads the CSV, builds the target, and does the outlier capping
import numpy as np                                          # numpy: arrays, random subsampling, and averaging results across seeds
import pandas.api.types as ptypes                           # ptypes: lets us reliably check "is this column already numeric?" across pandas versions
from sklearn.model_selection import train_test_split        # splits data into train/test partitions, one fresh split per seed
from sklearn.preprocessing import LabelEncoder               # converts each text column into integer codes (0, 1, 2, ...)
from sklearn.metrics import (                                # scoring tools used after every run
    accuracy_score,                                          #   overall fraction of correct predictions
    confusion_matrix,                                        #   3x3 table of actual vs. predicted class counts
    multilabel_confusion_matrix,                             #   per-class TP/TN/FP/FN breakdown (one-vs-rest)
    classification_report,                                   #   precision/recall/F1 per class
)
from skeLCS import eLCS                                      # skeLCS: the actual, unmodified eLCS implementation (same as task2_raw_lcs.py)


# -----------------------------------------------------------------------------
# SECTION 2: LOAD THE RAW DATASET
# -----------------------------------------------------------------------------

RAW_DATA_PATH = "ai_company_adoption.csv"                    # path to the raw CSV (adjust if it lives elsewhere)

df = pd.read_csv(RAW_DATA_PATH)                              # load the full raw file into a DataFrame

print("Raw dataset shape:", df.shape)                        # sanity check: should print (150000, 43)


# -----------------------------------------------------------------------------
# SECTION 3: BUILD THE CLASSIFICATION TARGET (QUANTILE-BASED CUTOFFS)
# -----------------------------------------------------------------------------
# Identical target definition to task2_raw_lcs.py and task4_improved_lcs.py --
# this MUST stay identical across every model in the Task 6 comparison, or
# the models would be solving different problems.

df["productivity_class"] = pd.qcut(                          # pd.qcut: splits a continuous column into equal-frequency bins
    df["productivity_change_percent"],                        # the continuous column we are converting into classes
    q=3,                                                       # 3 groups = tertiles (Low / Medium / High)
    labels=[0, 1, 2],                                          # numeric labels directly (0=Low, 1=Medium, 2=High)
).astype(int)                                                  # force the result to plain integers rather than pandas' categorical type

print(df["productivity_class"].value_counts())                # sanity check: should show ~50,000 rows per class


# -----------------------------------------------------------------------------
# SECTION 4: PREPROCESS THE DATASET (MISSINGNESS, DUPLICATES, LEAKAGE, OUTLIERS)
# -----------------------------------------------------------------------------

missing_counts = df.isnull().sum()                             # count missing values in every column
n_columns_with_missing = (missing_counts > 0).sum()             # how many columns have at least one missing value
print(f"Columns with missing values: {n_columns_with_missing} (out of {df.shape[1]})")  # report the finding

n_duplicate_rows = df.duplicated().sum()                        # count fully duplicated rows across the whole dataset
print(f"Fully duplicated rows: {n_duplicate_rows}")              # report the finding
# Both checks come back at zero for this dataset -- there is nothing to drop
# or impute here. We still run and report these checks explicitly, since
# confirming a clean dataset is itself a documented Task 3 data-quality step,
# not something to skip just because there was nothing to fix.

OUTCOME_FAMILY_COLUMNS = [                                       # columns that measure a similar "AI paid off" outcome as our target
    "revenue_growth_percent",
    "cost_reduction_percent",
    "customer_satisfaction",
    "time_saved_per_week",
    "employee_satisfaction_score",
    "jobs_created",
    "jobs_displaced",
    "reskilled_employees",
    "remote_work_percentage",
    "innovation_score",
]
ID_AND_TARGET_SOURCE_COLUMNS = [                                  # pure identifiers and the column our target was derived from
    "response_id",
    "company_id",
    "productivity_change_percent",
    "productivity_class",
]
DROP_COLUMNS = ID_AND_TARGET_SOURCE_COLUMNS + OUTCOME_FAMILY_COLUMNS  # everything excluded from this preprocessed feature set

feature_columns = [c for c in df.columns if c not in DROP_COLUMNS]  # every remaining column becomes a feature
print(f"Using {len(feature_columns)} preprocessed feature columns "  # sanity check: should print 29
      f"(vs. 40 in the raw baseline and 7 in the improved model).")

X_df = df[feature_columns].copy()                                 # build a working copy of just the feature columns

NUMERIC_COLUMNS_TO_CAP = [                                          # the 5 core numeric predictors we apply outlier capping to
    "ai_adoption_rate",
    "ai_maturity_score",
    "ai_failure_rate",
    "ai_training_hours",
    "task_automation_rate",
]
n_capped_total = 0                                                   # running count of how many values get capped, across all columns
for col in NUMERIC_COLUMNS_TO_CAP:                                    # loop over each numeric column we want to winsorize
    q1 = X_df[col].quantile(0.25)                                      # first quartile
    q3 = X_df[col].quantile(0.75)                                      # third quartile
    iqr = q3 - q1                                                       # interquartile range
    lower_bound = q1 - 1.5 * iqr                                        # standard IQR lower fence
    upper_bound = q3 + 1.5 * iqr                                        # standard IQR upper fence
    n_capped = ((X_df[col] < lower_bound) | (X_df[col] > upper_bound)).sum()  # how many values fall outside the fences
    n_capped_total += n_capped                                            # add to the running total
    X_df[col] = X_df[col].clip(lower=lower_bound, upper=upper_bound)       # cap (winsorize) values to the fence, rather than dropping the row
    print(f"  {col}: capped {n_capped} outlier values to [{lower_bound:.2f}, {upper_bound:.2f}]")  # report per-column

print(f"Total values capped across all numeric columns: {n_capped_total}")  # overall summary of the outlier-handling step

for col in X_df.columns:                                             # loop over every remaining feature column
    if not ptypes.is_numeric_dtype(X_df[col]):                        # check whether this column is already numeric
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))  # if it's text, convert it to integer codes (same basic approach as the raw baseline -- careful ordinal/one-hot encoding is reserved for the curated 7-column set in Task 4)

X = X_df.values.astype(float)                                        # convert the fully-numeric, cleaned DataFrame into a plain numpy array
y = df["productivity_class"].values.astype(int)                       # target vector (identical to task2_raw_lcs.py and task4_improved_lcs.py)

print("X shape:", X.shape, "| y shape:", y.shape)                      # sanity check: X should be (150000, 29)


# -----------------------------------------------------------------------------
# SECTION 5: REPEATED-EVALUATION SETTINGS (IDENTICAL TO THE OTHER TWO SCRIPTS)
# -----------------------------------------------------------------------------
# Every setting here is copied EXACTLY from task2_raw_lcs.py and
# task4_improved_lcs.py. Keeping these identical across all three eLCS
# scripts is what makes the three-way comparison in Task 6 valid -- only the
# feature set built in Sections 3-4 above changes between them.

N_SEEDS = 10                                                     # run the whole experiment 10 times, once per seed, as required
TRAIN_SUBSAMPLE_SIZE = 1000                                      # number of training rows actually used to fit eLCS in each run
TEST_EVAL_SIZE = 5000                                            # number of held-out test rows used to score each run
POPULATION_SIZE_N = round(1.5 * TRAIN_SUBSAMPLE_SIZE)             # eLCS population size, following the lecturer's "N ~ 1.5x training instances" guideline
LEARNING_ITERATIONS = 5 * TRAIN_SUBSAMPLE_SIZE                    # same iteration budget as the other two eLCS scripts

print(f"Population size N = {POPULATION_SIZE_N}, learning_iterations = {LEARNING_ITERATIONS}")  # confirm the derived settings


# -----------------------------------------------------------------------------
# SECTION 6: RUN THE 10-SEED EVALUATION LOOP
# -----------------------------------------------------------------------------
# Same seeds, same split logic, same subsampling logic as both other eLCS
# scripts -- this keeps all three LCS variants comparable on identical
# train/test row selections per seed.

per_seed_accuracy = []                                           # will hold one accuracy value per seed
all_true_labels = []                                              # will hold every true test label across all seeds, concatenated
all_predicted_labels = []                                         # will hold every predicted test label across all seeds, concatenated
last_model = None                                                 # keeps a reference to the final trained model, for exporting rules afterwards

for seed in range(N_SEEDS):                                       # loop once per random seed (0 through 9)
    X_train, X_test, y_train, y_test = train_test_split(            # split the FULL 150,000-row dataset for this seed
        X, y,                                                        # preprocessed feature matrix and target vector
        test_size=0.2,                                               # 80/20 train/test split, matching both other eLCS scripts
        random_state=seed,                                           # a different split for every seed, but reproducible if re-run
        stratify=y,                                                   # keep Low/Medium/High proportions balanced in both partitions
    )

    rng = np.random.RandomState(seed)                                # a seeded random generator, used only for the subsampling step below
    train_subsample_idx = rng.choice(                                 # randomly pick which training rows to actually use this run
        len(X_train), TRAIN_SUBSAMPLE_SIZE, replace=False               # pick TRAIN_SUBSAMPLE_SIZE unique row positions from the training partition
    )
    X_train_sub = X_train[train_subsample_idx]                         # the actual training features used this run
    y_train_sub = y_train[train_subsample_idx]                         # the actual training labels used this run

    X_test_eval = X_test[:TEST_EVAL_SIZE]                              # take a fixed-size slice of the test partition to keep evaluation fast
    y_test_eval = y_test[:TEST_EVAL_SIZE]                              # matching true labels for that slice

    model = eLCS(                                                      # instantiate a fresh, unmodified eLCS model for this seed
        learning_iterations=LEARNING_ITERATIONS,                        # how many training instances the algorithm samples while evolving rules
        N=POPULATION_SIZE_N,                                            # maximum number of rules the population can hold, per the lecturer's guideline
        random_state=seed,                                              # ties eLCS's own internal randomness to this seed too
    )
    model.fit(X_train_sub, y_train_sub)                                  # train the model on this seed's subsample

    y_pred = model.predict(X_test_eval)                                  # predict classes for this seed's test slice
    acc = accuracy_score(y_test_eval, y_pred)                             # compute this seed's accuracy
    per_seed_accuracy.append(acc)                                         # store it for later averaging

    all_true_labels.extend(y_test_eval.tolist())                          # add this seed's true labels to the running combined list
    all_predicted_labels.extend(y_pred.tolist())                          # add this seed's predictions to the running combined list

    last_model = model                                                    # keep the most recent model around (used for rule export in Section 9)

    print(f"Seed {seed}: accuracy = {acc:.4f}")                           # progress update after every seed


# -----------------------------------------------------------------------------
# SECTION 7: SUMMARISE ACCURACY ACROSS ALL 10 SEEDS
# -----------------------------------------------------------------------------

mean_accuracy = np.mean(per_seed_accuracy)                        # average accuracy across the 10 runs
std_accuracy = np.std(per_seed_accuracy)                          # standard deviation, showing how much results varied run-to-run

print(f"\nPreprocessed (unmodified) eLCS -- mean accuracy over {N_SEEDS} seeds: {mean_accuracy:.4f} (std {std_accuracy:.4f})")  # headline result for this row
print("Compare this against task2_raw_lcs.py's mean accuracy (effect of cleaning alone) and")  # reminder for the write-up
print("task4_improved_lcs.py's mean accuracy (added effect of feature selection) for the Task 6 table.")


# -----------------------------------------------------------------------------
# SECTION 8: CONFUSION MATRIX AND PER-CLASS TP / TN / FP / FN
# -----------------------------------------------------------------------------
# Same pooling approach as both other eLCS scripts: combine all 10 seeds'
# test predictions (10 x 5,000 = 50,000 evaluated rows) before computing the
# confusion matrix.

all_true_labels = np.array(all_true_labels)                        # convert the combined true-label list into a numpy array
all_predicted_labels = np.array(all_predicted_labels)               # convert the combined predicted-label list into a numpy array

overall_confusion = confusion_matrix(all_true_labels, all_predicted_labels)  # 3x3 matrix: rows = actual class, columns = predicted class
print("\nPooled confusion matrix across all 10 seeds (rows = actual, columns = predicted):")  # explain how to read it
print(overall_confusion)                                             # print the raw 3x3 matrix

print("\nClassification report (pooled across all seeds):")           # header for the detailed report
print(classification_report(                                          # precision/recall/F1 per class, computed on the pooled results
    all_true_labels, all_predicted_labels,
    target_names=["Low", "Medium", "High"],
    zero_division=0,                                                    # avoids noisy warnings when a class is never predicted
))

per_class_matrices = multilabel_confusion_matrix(all_true_labels, all_predicted_labels)  # one 2x2 [[TN,FP],[FN,TP]] matrix per class, one-vs-rest
class_names = ["Low", "Medium", "High"]                                # readable names matching class codes 0, 1, 2

print("\nPer-class TP / TN / FP / FN (one-vs-rest):")                  # explain what follows
for class_name, matrix in zip(class_names, per_class_matrices):          # loop through each class's 2x2 matrix
    tn, fp, fn, tp = matrix.ravel()                                        # unpack the 2x2 matrix into its four named quadrants
    print(                                                                  # print a readable one-line summary per class
        f"  {class_name:7s}: TP={tp:5d}  TN={tn:5d}  FP={fp:5d}  FN={fn:5d}  "
        f"-> means, for '{class_name}' vs. everything else: {tp} correctly flagged as {class_name}, "
        f"{tn} correctly flagged as NOT {class_name}, {fp} wrongly flagged as {class_name}, "
        f"{fn} {class_name} cases the model missed."
    )


# -----------------------------------------------------------------------------
# SECTION 9: EXPORT THE FINAL RULE POPULATION (FOR THE TASK 7 DISCUSSION)
# -----------------------------------------------------------------------------
# Exporting this row's rules lets Task 7 show all three LCS variants' rules
# side by side: raw (noisy, many columns), preprocessed (cleaned, still many
# columns), improved (cleaned AND curated to 7 columns).

last_model.export_final_rule_population(
    headerNames=np.array(feature_columns),                            # tell eLCS the real name of every one of our ~29 feature columns
    className="productivity_class",                                    # name of the target column in the exported file
    filename="task3_preprocessed_lcs_rules.csv",                        # output filename
    DCAL=True,                                                          # export in the readable "Detailed Compact Attribute List" format
)

rule_population = pd.read_csv("task3_preprocessed_lcs_rules.csv")       # read the exported rules back in
print(f"\nExported {len(rule_population)} rules to task3_preprocessed_lcs_rules.csv")  # confirm how many rules were saved
print(rule_population.sort_values("Numerosity", ascending=False).head(5))     # preview the 5 most common rules


# -----------------------------------------------------------------------------
# SECTION 10: SAVE RESULTS FOR THE TASK 5/6 COMPARISON
# -----------------------------------------------------------------------------
# Same file structure as the other two eLCS scripts (and the Random Forest
# script), so all the seed-result CSVs can be concatenated directly for the
# Task 5 statistical test and the Task 6 comparison table.

seed_results = pd.DataFrame({                                          # one row per seed, for later statistical comparison
    "seed": list(range(N_SEEDS)),
    "model": "Original eLCS (preprocessed, ~29 columns, leakage & outliers handled)",
    "accuracy": per_seed_accuracy,
})
seed_results.to_csv("task3_preprocessed_lcs_seed_results.csv", index=False)  # save the per-seed results to their own CSV

summary_row = pd.DataFrame([{                                          # one summary row, for quick reference
    "model": "Original eLCS (preprocessed, ~29 columns, leakage & outliers handled)",
    "mean_accuracy": mean_accuracy,
    "std_accuracy": std_accuracy,
    "n_seeds": N_SEEDS,
    "n_features": X.shape[1],
    "n_rules_last_seed": len(rule_population),
}])
summary_row.to_csv("task3_preprocessed_lcs_results.csv", index=False)   # save the summary row to its own CSV

print("\nSaved per-seed results to task3_preprocessed_lcs_seed_results.csv and summary to task3_preprocessed_lcs_results.csv")  # confirm the save
print(summary_row)                                                       # display the summary row in the output too
