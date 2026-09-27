# =============================================================================
# TASK 6: RANDOM FOREST -- CONVENTIONAL ML MODEL FOR THE COMPARISON TABLE
# -----------------------------------------------------------------------------
# This is one of the 3+ conventional (non-deep-learning) models required for
# Task 6's model comparison, alongside the Task 2 raw eLCS baseline and the
# Task 4 improved eLCS model. Random Forest is this team member's assigned
# model; teammates are separately building Decision Tree and Logistic
# Regression using the same structure as this file.
#
# To keep the comparison fair, this script deliberately mirrors the Task 4
# improved eLCS script wherever it matters for comparability:
#   - the SAME 3-class target (Low/Medium/High, via the same quantile cutoffs)
#   - the SAME 7 selected feature columns, with the SAME categorical encoding
#   - the SAME 10 random seeds, the SAME 80/20 stratified split per seed, and
#     the SAME fixed-size test evaluation slice (5,000 rows)
# This is what makes a valid, paired, apples-to-apples comparison possible in
# Task 5's statistical test -- every model is scored on the exact same test
# rows for each seed.
#
# ONE DELIBERATE DIFFERENCE, worth calling out in the report: unlike eLCS,
# Random Forest is trained on the FULL training partition (~120,000 rows) for
# each seed, not a small subsample. eLCS was limited to a small subsample
# purely because of eLCS's own computational cost (population size x
# iterations); Random Forest has no such limitation and training on the full
# partition reflects realistic best practice for this algorithm. This means
# raw accuracy numbers shouldn't be read as "same training budget for every
# model" -- but the paired same-test-set evaluation per seed still supports a
# valid statistical comparison of final test performance, which is what
# Task 5 actually needs.
#
# Random Forest also needs NO feature scaling (unlike Logistic Regression) --
# it's a tree-based method, so it splits on raw feature values directly and
# is unaffected by different features having different numeric ranges.
#
# NOTE FOR THE TEAM: every line below has a comment explaining what it does.
# Read through this and rewrite the comments in your own words before you
# submit -- the lecturer wants to see that you understand the code, not just
# that it runs.
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import pandas as pd                                        # pandas: loads the CSV and builds the quantile-binned target
import numpy as np                                          # numpy: arrays and averaging results across seeds
from sklearn.model_selection import train_test_split        # splits data into train/test partitions, one fresh split per seed
from sklearn.preprocessing import LabelEncoder               # converts each text column into integer codes (0, 1, 2, ...)
from sklearn.ensemble import RandomForestClassifier           # the actual Random Forest model: an ensemble of many decision trees
from sklearn.metrics import (                                # scoring tools used after every run
    accuracy_score,                                          #   overall fraction of correct predictions
    confusion_matrix,                                        #   3x3 table of actual vs. predicted class counts
    multilabel_confusion_matrix,                             #   per-class TP/TN/FP/FN breakdown (one-vs-rest)
    classification_report,                                   #   precision/recall/F1 per class
)


# -----------------------------------------------------------------------------
# SECTION 2: LOAD THE RAW DATASET
# -----------------------------------------------------------------------------

RAW_DATA_PATH = "ai_company_adoption.csv"                    # path to the raw CSV (adjust if it lives elsewhere)

df = pd.read_csv(RAW_DATA_PATH)                              # load the full raw file into a DataFrame

print("Raw dataset shape:", df.shape)                        # sanity check: should print (150000, 43)


# -----------------------------------------------------------------------------
# SECTION 3: BUILD THE CLASSIFICATION TARGET (QUANTILE-BASED CUTOFFS)
# -----------------------------------------------------------------------------
# Identical target definition to both eLCS scripts -- this MUST stay
# identical across every model in the comparison, or the models would be
# solving different problems and the comparison in Task 6 would be invalid.

df["productivity_class"] = pd.qcut(                          # pd.qcut: splits a continuous column into equal-frequency bins
    df["productivity_change_percent"],                        # the continuous column we are converting into classes
    q=3,                                                       # 3 groups = tertiles (Low / Medium / High)
    labels=[0, 1, 2],                                          # numeric labels directly (0=Low, 1=Medium, 2=High)
).astype(int)                                                  # force the result to plain integers rather than pandas' categorical type

print(df["productivity_class"].value_counts())                # sanity check: should show ~50,000 rows per class


# -----------------------------------------------------------------------------
# SECTION 4: BUILD THE FEATURE MATRIX -- SAME 7 SELECTED COLUMNS AS TASK 4
# -----------------------------------------------------------------------------
# Identical feature set and encoding to the Task 4 improved eLCS script, so
# that any performance difference between Random Forest and Improved eLCS
# reflects the ALGORITHM, not a difference in the data each one was given.

SELECTED_FEATURE_COLUMNS = [                                    # the team's shortlisted columns (same as Task 4):
    "ai_adoption_rate",        # r = 0.67 with the target -- the variable our whole Phase 1 report was built around
    "ai_maturity_score",       # r = 0.74 -- the single strongest numeric predictor in the dataset
    "ai_failure_rate",         # r = -0.59 -- the strongest negative predictor
    "ai_training_hours",       # r = 0.63 -- whether staff were actually enabled to use AI well
    "task_automation_rate",    # r = 0.58 -- how much of the workflow is actually automated
    "ai_adoption_stage",       # categorical (none/pilot/partial/full) -- cleanest signal of any column
    "industry",                # categorical (9 sectors) -- weaker but real signal
]

X_df = df[SELECTED_FEATURE_COLUMNS].copy()                       # build a working copy of just these 7 columns

CATEGORICAL_COLUMNS = ["ai_adoption_stage", "industry"]           # the two text columns among our 7 selected features
for col in CATEGORICAL_COLUMNS:                                    # loop over just these two columns
    X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))  # convert each one into integer codes
    # Random Forest splits a tree node on rules like "is this feature's code
    # <= 3?", so it doesn't need the categories to have a meaningful numeric
    # order the way a linear model would -- integer codes are sufficient
    # here, unlike for Logistic Regression, which would need one-hot
    # encoding instead to avoid implying a false ordering between categories.

X = X_df.values.astype(float)                                     # convert the fully-numeric DataFrame into a plain numpy array of floats
y = df["productivity_class"].values.astype(int)                   # target vector as a numpy array of integers (identical to both eLCS scripts)

print("X shape:", X.shape, "| y shape:", y.shape)                  # sanity check: X should be (150000, 7)


# -----------------------------------------------------------------------------
# SECTION 5: REPEATED-EVALUATION SETTINGS
# -----------------------------------------------------------------------------
# N_SEEDS and TEST_EVAL_SIZE match both eLCS scripts exactly, so every model
# in the Task 6 comparison is scored on the same number of test rows per
# seed. There is no TRAIN_SUBSAMPLE_SIZE here, since Random Forest trains on
# the FULL training partition for each seed (see the note at the top of this
# file for why that's a deliberate, documented difference from eLCS).

N_SEEDS = 10                                                      # run the whole experiment 10 times, once per seed, matching both eLCS scripts
TEST_EVAL_SIZE = 5000                                             # number of held-out test rows used to score each run (matches both eLCS scripts)
N_ESTIMATORS = 100                                                # number of individual decision trees in the forest
MAX_DEPTH = 12                                                    # cap on how deep each tree can grow, to control overfitting and keep training fast

print(f"Random Forest settings: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")  # confirm the settings


# -----------------------------------------------------------------------------
# SECTION 6: RUN THE 10-SEED EVALUATION LOOP
# -----------------------------------------------------------------------------
# Because this script uses the SAME dataframe row order, the SAME y target,
# and the SAME seeds as both eLCS scripts, train_test_split selects the exact
# same rows for training and testing at every seed as those scripts did --
# this is what makes the comparison in Task 5/6 a fair, paired one.

per_seed_accuracy = []                                            # will hold one accuracy value per seed
all_true_labels = []                                               # will hold every true test label across all seeds, concatenated
all_predicted_labels = []                                          # will hold every predicted test label across all seeds, concatenated
last_model = None                                                  # keeps a reference to the final trained model, for feature importances afterwards

for seed in range(N_SEEDS):                                        # loop once per random seed (0 through 9)
    X_train, X_test, y_train, y_test = train_test_split(             # split the FULL 150,000-row dataset for this seed
        X, y,                                                         # feature matrix and target vector
        test_size=0.2,                                                # 80/20 train/test split, matching both eLCS scripts
        random_state=seed,                                            # a different split for every seed, but reproducible if re-run
        stratify=y,                                                    # keep Low/Medium/High proportions balanced in both partitions
    )

    X_test_eval = X_test[:TEST_EVAL_SIZE]                             # take the same fixed-size slice of the test partition used by both eLCS scripts
    y_test_eval = y_test[:TEST_EVAL_SIZE]                             # matching true labels for that slice

    model = RandomForestClassifier(                                    # instantiate a fresh Random Forest for this seed
        n_estimators=N_ESTIMATORS,                                      # how many trees to grow and average over
        max_depth=MAX_DEPTH,                                            # limits each tree's depth to reduce overfitting
        random_state=seed,                                              # ties the forest's internal randomness to this seed too
        n_jobs=-1,                                                      # use all available CPU cores to train faster
    )
    model.fit(X_train, y_train)                                          # train on the FULL training partition for this seed (see Section 5 note)

    y_pred = model.predict(X_test_eval)                                  # predict classes for this seed's test slice
    acc = accuracy_score(y_test_eval, y_pred)                             # compute this seed's accuracy
    per_seed_accuracy.append(acc)                                         # store it for later averaging

    all_true_labels.extend(y_test_eval.tolist())                          # add this seed's true labels to the running combined list
    all_predicted_labels.extend(y_pred.tolist())                          # add this seed's predictions to the running combined list

    last_model = model                                                    # keep the most recent model around (used for feature importances in Section 9)

    print(f"Seed {seed}: accuracy = {acc:.4f}")                           # progress update after every seed


# -----------------------------------------------------------------------------
# SECTION 7: SUMMARISE ACCURACY ACROSS ALL 10 SEEDS
# -----------------------------------------------------------------------------

mean_accuracy = np.mean(per_seed_accuracy)                        # average accuracy across the 10 runs
std_accuracy = np.std(per_seed_accuracy)                          # standard deviation, showing how much results varied run-to-run

print(f"\nRandom Forest -- mean accuracy over {N_SEEDS} seeds: {mean_accuracy:.4f} (std {std_accuracy:.4f})")  # headline result for Task 6
print("Compare this against the raw eLCS baseline and improved eLCS mean accuracies for the Task 6 table.")     # reminder for the write-up


# -----------------------------------------------------------------------------
# SECTION 8: CONFUSION MATRIX AND PER-CLASS TP / TN / FP / FN
# -----------------------------------------------------------------------------
# Same pooling approach as both eLCS scripts: combine all 10 seeds' test
# predictions (10 x 5,000 = 50,000 evaluated rows) before computing the
# confusion matrix, giving a much more stable picture than any single seed.

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
# SECTION 9: FEATURE IMPORTANCES (RANDOM FOREST'S EQUIVALENT OF LCS RULES)
# -----------------------------------------------------------------------------
# Random Forest doesn't produce human-readable IF/THEN rules like eLCS does,
# but it does provide feature importances -- a ranking of which columns the
# forest relied on most heavily to make its splits. This is the natural
# comparison point for the Task 7 interpretability discussion: eLCS gives you
# specific readable rules, Random Forest gives you a ranked list of "what
# mattered", but neither is a black box in the same way a deep neural network
# would be.

importances = pd.DataFrame({                                       # build a small table pairing each feature with its importance score
    "feature": SELECTED_FEATURE_COLUMNS,
    "importance": last_model.feature_importances_,
}).sort_values("importance", ascending=False)                        # sort so the most influential feature appears first

print("\nFeature importances (from the final seed's trained forest):")  # header
print(importances.to_string(index=False))                            # print the full ranked table

importances.to_csv("task6_random_forest_feature_importances.csv", index=False)  # save for use in the Task 7 write-up
print("\nSaved feature importances to task6_random_forest_feature_importances.csv")  # confirm the save


# -----------------------------------------------------------------------------
# SECTION 10: SAVE RESULTS FOR THE TASK 5/6 COMPARISON
# -----------------------------------------------------------------------------
# Same file structure as both eLCS scripts, so all three CSVs can be
# concatenated directly for the Task 5 statistical test and the Task 6
# comparison table.

seed_results = pd.DataFrame({                                          # one row per seed, for later statistical comparison
    "seed": list(range(N_SEEDS)),
    "model": "Random Forest (7 selected features)",
    "accuracy": per_seed_accuracy,
})
seed_results.to_csv("task6_random_forest_seed_results.csv", index=False)  # save the per-seed results to their own CSV

summary_row = pd.DataFrame([{                                          # one summary row, for quick reference
    "model": "Random Forest (7 selected features)",
    "mean_accuracy": mean_accuracy,
    "std_accuracy": std_accuracy,
    "n_seeds": N_SEEDS,
    "n_features": X.shape[1],
    "n_rules_last_seed": None,                                           # Random Forest has no rule count; kept as a column so this CSV lines up with the eLCS ones
}])
summary_row.to_csv("task6_random_forest_results.csv", index=False)      # save the summary row to its own CSV

print("\nSaved per-seed results to task6_random_forest_seed_results.csv and summary to task6_random_forest_results.csv")  # confirm the save
print(summary_row)                                                       # display the summary row in the output too
