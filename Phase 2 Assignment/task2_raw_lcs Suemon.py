# =============================================================================
# TASK 2: ORIGINAL (UNMODIFIED) eLCS SYSTEM ON THE RAW DATASET
# -----------------------------------------------------------------------------
# Per the lecturer's clarification: this baseline uses the RAW dataset with
# EVERY usable column (not just ai_adoption_rate), converting any non-numeric
# column into numbers. No feature selection, no cleaning, no justification of
# which columns matter -- that thoughtful work is reserved for Task 3/4. The
# lecturer expects this version to perform noticeably worse than the Task 4
# improved model; that gap IS the evidence for Task 4's improvement claim.
#
# NOTE FOR THE TEAM: every line below has a comment explaining what it does.
# Read through this and rewrite the comments in your own words before you
# submit -- the lecturer wants to see that you understand the code, not just
# that it runs.
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import pandas as pd                                        # pandas: loads the CSV and builds the quantile-based target
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
from skeLCS import eLCS                                      # skeLCS: the actual, unmodified eLCS implementation


# -----------------------------------------------------------------------------
# SECTION 2: LOAD THE RAW DATASET
# -----------------------------------------------------------------------------

RAW_DATA_PATH = "ai_company_adoption.csv"                    # path to the raw CSV (adjust if it lives elsewhere)

df = pd.read_csv(RAW_DATA_PATH)                              # load the full raw file into a DataFrame

print("Raw dataset shape:", df.shape)                        # sanity check: should print (150000, 43)


# -----------------------------------------------------------------------------
# SECTION 3: BUILD THE CLASSIFICATION TARGET (QUANTILE-BASED CUTOFFS)
# -----------------------------------------------------------------------------
# Same target definition used throughout Phase 2: eLCS needs a discrete class,
# so productivity_change_percent is split into three EQUALLY SIZED groups
# (quantiles/tertiles) rather than fixed-width bins, avoiding the severe class
# imbalance fixed-width bins would create given how skewed the raw percentage
# values are.

df["productivity_class"] = pd.qcut(                          # pd.qcut: splits a continuous column into equal-frequency bins
    df["productivity_change_percent"],                        # the continuous column we are converting into classes
    q=3,                                                       # 3 groups = tertiles (Low / Medium / High)
    labels=[0, 1, 2],                                          # numeric labels directly, since eLCS needs numeric classes (0=Low, 1=Medium, 2=High)
).astype(int)                                                  # force the result to plain integers rather than pandas' categorical type

print(df["productivity_class"].value_counts())                # sanity check: should show ~50,000 rows per class


# -----------------------------------------------------------------------------
# SECTION 4: BUILD THE RAW FEATURE MATRIX -- EVERY COLUMN, NUMBERS ONLY
# -----------------------------------------------------------------------------
# We deliberately keep almost every column here, since this run is meant to
# represent the "naive, no feature engineering" baseline.
#
# Two columns are still dropped, but for a different reason than feature
# selection -- they are pure identifiers, not real-world measurements:
#   - response_id: a unique row number (150,000 distinct values, one per row)
#   - company_id: a unique company code (10,000 distinct values)
# Converting an identifier into a number and feeding it to a model doesn't
# represent "using the raw data" -- it would just hand the model a random
# lookup key with no real relationship to productivity, which isn't what
# "unmodified raw baseline" is meant to test.
#
# productivity_change_percent itself is also dropped, because it is the exact
# column our target was derived from -- keeping it in would let the model see
# the answer directly (data leakage), which is a genuine bug, not just "bad
# practice", so it is excluded even from this deliberately naive baseline.

ID_AND_TARGET_SOURCE_COLUMNS = [                              # columns to exclude and why:
    "response_id",                                             #   pure row identifier, not a real feature
    "company_id",                                               #   pure company identifier, not a real feature
    "productivity_change_percent",                              #   this is what our target was derived from -- must exclude to avoid leakage
    "productivity_class",                                        #   the target itself, obviously not a predictor
]

feature_columns = [c for c in df.columns if c not in ID_AND_TARGET_SOURCE_COLUMNS]  # every remaining column becomes a feature (40 columns)
print(f"Using {len(feature_columns)} raw feature columns.")   # sanity check: should print 40

X_raw_df = df[feature_columns].copy()                          # build a working copy of just the feature columns, so we don't modify df itself

for col in X_raw_df.columns:                                    # loop over every feature column
    if not ptypes.is_numeric_dtype(X_raw_df[col]):               # check whether this column is already numeric
        X_raw_df[col] = LabelEncoder().fit_transform(             # if it's text (e.g. "industry", "country", "quarter"), convert it to integer codes
            X_raw_df[col].astype(str)                              # force to string first, in case of mixed types
        )
        # NOTE: LabelEncoder assigns arbitrary integer codes (e.g. Finance=0,
        # Healthcare=1, Technology=2 ...) with no meaningful order between
        # them. eLCS will then treat any such column with more than 10
        # distinct values as if it were a continuous number line, which is
        # not a meaningful way to represent a category like "country". This
        # is a real limitation of the naive/raw approach, and it's exactly
        # the kind of issue Task 3/4's more careful encoding is meant to fix.

X = X_raw_df.values.astype(float)                               # convert the fully-numeric DataFrame into a plain numpy array of floats
y = df["productivity_class"].values.astype(int)                 # target vector as a numpy array of integers

print("X shape:", X.shape, "| y shape:", y.shape)                # sanity check: X should be (150000, 40)


# -----------------------------------------------------------------------------
# SECTION 5: REPEATED-EVALUATION SETTINGS
# -----------------------------------------------------------------------------
# The lecturer asked for the model to be evaluated at least 10 times, each
# time with a different random seed, and the results averaged -- this makes
# our accuracy estimate robust to the randomness inside both the train/test
# split and eLCS's own genetic algorithm, rather than reporting one lucky (or
# unlucky) run.
#
# We also can't train eLCS on the full 120,000-row training partition in a
# reasonable time on a student laptop, so each run trains on a fixed-size
# random SUBSAMPLE of the training partition. The lecturer's guideline is
# that eLCS's population size N should be roughly 1.5x the number of training
# instances used, so N scales directly off TRAIN_SUBSAMPLE_SIZE below.

N_SEEDS = 10                                                     # run the whole experiment 10 times, once per seed, as required
TRAIN_SUBSAMPLE_SIZE = 1000                                      # number of training rows actually used to fit eLCS in each run
TEST_EVAL_SIZE = 5000                                            # number of held-out test rows used to score each run (kept fixed for speed)
POPULATION_SIZE_N = round(1.5 * TRAIN_SUBSAMPLE_SIZE)             # eLCS population size, following the lecturer's "N ~ 1.5x training instances" guideline
LEARNING_ITERATIONS = 5 * TRAIN_SUBSAMPLE_SIZE                    # each training instance gets sampled ~5 times on average during evolution

print(f"Population size N = {POPULATION_SIZE_N}, learning_iterations = {LEARNING_ITERATIONS}")  # confirm the derived settings


# -----------------------------------------------------------------------------
# SECTION 6: RUN THE 10-SEED EVALUATION LOOP
# -----------------------------------------------------------------------------
# IMPORTANT: for every seed, we split the FULL dataset the same way we would
# for any model (80/20, stratified), then subsample the training side. Using
# the same seed and the same underlying row order as the Task 4 improved
# model guarantees both models are compared on IDENTICAL train/test row
# selections -- this is what makes a fair, paired comparison possible later
# in Task 5's statistical test.

per_seed_accuracy = []                                           # will hold one accuracy value per seed
all_true_labels = []                                              # will hold every true test label across all seeds, concatenated
all_predicted_labels = []                                         # will hold every predicted test label across all seeds, concatenated
last_model = None                                                 # keeps a reference to the final trained model, for exporting rules afterwards

for seed in range(N_SEEDS):                                       # loop once per random seed (0 through 9)
    X_train, X_test, y_train, y_test = train_test_split(            # split the FULL 150,000-row dataset for this seed
        X, y,                                                        # full raw feature matrix and target vector
        test_size=0.2,                                               # 80/20 train/test split, same convention used throughout the project
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

print(f"\nRaw baseline eLCS -- mean accuracy over {N_SEEDS} seeds: {mean_accuracy:.4f} (std {std_accuracy:.4f})")  # headline result for Task 2


# -----------------------------------------------------------------------------
# SECTION 8: CONFUSION MATRIX AND PER-CLASS TP / TN / FP / FN
# -----------------------------------------------------------------------------
# We combine every seed's test predictions into one large pooled set (10 x
# 5,000 = 50,000 evaluated rows in total) before computing the confusion
# matrix, giving a much more stable picture than any single seed's 5,000-row
# result would.

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
# We export the rule population from the LAST seed's trained model as a
# representative example of what the raw baseline's rules look like -- useful
# for Task 7's comparison against the Task 4 improved model's rules.

last_model.export_final_rule_population(
    headerNames=np.array(feature_columns),                            # tell eLCS the real name of every one of our 40 feature columns
    className="productivity_class",                                    # name of the target column in the exported file
    filename="task2_raw_lcs_rules.csv",                                 # output filename
    DCAL=True,                                                          # export in the readable "Detailed Compact Attribute List" format
)

rule_population = pd.read_csv("task2_raw_lcs_rules.csv")                # read the exported rules back in
print(f"\nExported {len(rule_population)} rules to task2_raw_lcs_rules.csv")  # confirm how many rules were saved
print(rule_population.sort_values("Numerosity", ascending=False).head(5))     # preview the 5 most common rules


# -----------------------------------------------------------------------------
# SECTION 10: SAVE RESULTS FOR THE TASK 5/6 COMPARISON
# -----------------------------------------------------------------------------
# We save every seed's individual accuracy (not just the average) because
# Task 5's statistical test needs the full list of per-seed scores, not just
# a single summary number.

seed_results = pd.DataFrame({                                          # one row per seed, for later statistical comparison
    "seed": list(range(N_SEEDS)),
    "model": "Original eLCS (raw, all 40 numeric-encoded columns)",
    "accuracy": per_seed_accuracy,
})
seed_results.to_csv("task2_raw_lcs_seed_results.csv", index=False)      # save the per-seed results to their own CSV

summary_row = pd.DataFrame([{                                          # one summary row, for quick reference
    "model": "Original eLCS (raw, all 40 numeric-encoded columns)",
    "mean_accuracy": mean_accuracy,
    "std_accuracy": std_accuracy,
    "n_seeds": N_SEEDS,
    "n_features": X.shape[1],
    "n_rules_last_seed": len(rule_population),
}])
summary_row.to_csv("task2_raw_lcs_results.csv", index=False)            # save the summary row to its own CSV

print("\nSaved per-seed results to task2_raw_lcs_seed_results.csv and summary to task2_raw_lcs_results.csv")  # confirm the save
print(summary_row)                                                       # display the summary row in the output too
