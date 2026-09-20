# =============================================================================
# TASK 4: IMPROVED eLCS SYSTEM ON THE SELECTED FEATURE SET
# -----------------------------------------------------------------------------
# This is the improved version of the Task 2 raw baseline. The IMPROVEMENT
# being tested here is deliberately just ONE thing: feature selection.
# Everything else -- population size N, learning_iterations, the train/test
# split logic, the number of seeds evaluated -- is kept IDENTICAL to Task 2's
# raw script. That means any accuracy difference between the two scripts can
# be attributed specifically to the feature set, rather than being muddied by
# also changing several things at once.
#
# The 7 columns used here (ai_adoption_rate, ai_maturity_score,
# ai_failure_rate, ai_training_hours, task_automation_rate,
# ai_adoption_stage, industry) were chosen based on correlation strength with
# productivity_change_percent and group-mean separation -- see the team's
# earlier feature-shortlisting discussion for the full justification.
#
# "REDUCED SEARCH SPACE" (a term the lecturer specifically asked us to
# explain in the report): every eLCS rule is an IF/THEN condition built out
# of the available features. With 40 raw columns, there are vastly more
# possible ways to combine feature-ranges into a rule than with 7 curated
# columns -- most of those extra combinations in the 40-column version are
# just noise (e.g. combinations involving country codes or arbitrary ID-like
# columns). Cutting down to 7 informative columns shrinks the space of
# possible rules the genetic algorithm has to search through, which is why
# the SAME population size and SAME number of iterations can find much
# better rules here than in the raw baseline -- the search is working over a
# much smaller, more relevant space.
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
from sklearn.model_selection import train_test_split        # splits data into train/test partitions, one fresh split per seed
from sklearn.preprocessing import LabelEncoder               # converts each text column into integer codes (0, 1, 2, ...)
from sklearn.metrics import (                                # scoring tools used after every run
    accuracy_score,                                          #   overall fraction of correct predictions
    confusion_matrix,                                        #   3x3 table of actual vs. predicted class counts
    multilabel_confusion_matrix,                             #   per-class TP/TN/FP/FN breakdown (one-vs-rest)
    classification_report,                                   #   precision/recall/F1 per class
)
from skeLCS import eLCS                                      # skeLCS: the actual, unmodified eLCS implementation (the algorithm itself is NOT changed -- only its inputs are)


# -----------------------------------------------------------------------------
# SECTION 2: LOAD THE RAW DATASET
# -----------------------------------------------------------------------------

RAW_DATA_PATH = "ai_company_adoption.csv"                    # path to the raw CSV (adjust if it lives elsewhere)

df = pd.read_csv(RAW_DATA_PATH)                              # load the full raw file into a DataFrame

print("Raw dataset shape:", df.shape)                        # sanity check: should print (150000, 43)


# -----------------------------------------------------------------------------
# SECTION 3: BUILD THE CLASSIFICATION TARGET (QUANTILE-BASED CUTOFFS)
# -----------------------------------------------------------------------------
# Identical target definition to Task 2's raw script, and this MUST stay
# identical -- if the class boundaries were different between the two
# scripts, any accuracy difference we report wouldn't be a fair comparison,
# since the two models would effectively be solving different problems.

df["productivity_class"] = pd.qcut(                          # pd.qcut: splits a continuous column into equal-frequency bins
    df["productivity_change_percent"],                        # the continuous column we are converting into classes
    q=3,                                                       # 3 groups = tertiles (Low / Medium / High)
    labels=[0, 1, 2],                                          # numeric labels directly (0=Low, 1=Medium, 2=High)
).astype(int)                                                  # force the result to plain integers rather than pandas' categorical type

print(df["productivity_class"].value_counts())                # sanity check: should show ~50,000 rows per class


# -----------------------------------------------------------------------------
# SECTION 4: BUILD THE IMPROVED FEATURE MATRIX -- 7 SELECTED COLUMNS
# -----------------------------------------------------------------------------
# Unlike Task 2's raw script (which numeric-encodes all 40 remaining
# columns), here we hand-pick just 7 columns with strong, justified
# relationships to productivity_change_percent. Two of them are categorical
# and need encoding; the other five are already numeric.

SELECTED_FEATURE_COLUMNS = [                                    # the team's shortlisted columns, with the correlation evidence behind each:
    "ai_adoption_rate",        # r = 0.67 with the target -- the variable our whole Phase 1 report was built around
    "ai_maturity_score",       # r = 0.74 -- the single strongest numeric predictor in the dataset
    "ai_failure_rate",         # r = -0.59 -- the strongest negative predictor (more failures, less productivity gain)
    "ai_training_hours",       # r = 0.63 -- captures whether staff were actually enabled to use AI well
    "task_automation_rate",    # r = 0.58 -- how much of the workflow is actually automated, not just adopted
    "ai_adoption_stage",       # categorical (none/pilot/partial/full) -- cleanest signal of any column: group means span 2.4% to 19.8%
    "industry",                # categorical (9 sectors) -- weaker but real signal: group means span 8.6% to 11.2%
]

X_improved_df = df[SELECTED_FEATURE_COLUMNS].copy()             # build a working copy of just these 7 columns

CATEGORICAL_COLUMNS = ["ai_adoption_stage", "industry"]          # the two text columns among our 7 selected features
for col in CATEGORICAL_COLUMNS:                                   # loop over just these two columns
    X_improved_df[col] = LabelEncoder().fit_transform(              # convert each one into integer codes
        X_improved_df[col].astype(str)                               # force to string first, for safety
    )
    # Both of these columns have a SMALL number of distinct values
    # (ai_adoption_stage has 4, industry has 9), both under eLCS's default
    # discrete_attribute_limit of 10. This means eLCS will automatically
    # treat them as DISCRETE/nominal attributes (matched by exact equality)
    # rather than as a continuous number line -- unlike what happened to
    # high-cardinality columns such as "country" in the raw Task 2 baseline.
    # This is a real, concrete piece of "why the improved preprocessing
    # helps eLCS" for the report.

X = X_improved_df.values.astype(float)                           # convert the fully-numeric DataFrame into a plain numpy array of floats
y = df["productivity_class"].values.astype(int)                  # target vector as a numpy array of integers (identical to Task 2's y)

print("X shape:", X.shape, "| y shape:", y.shape)                 # sanity check: X should be (150000, 7) -- versus (150000, 40) in Task 2


# -----------------------------------------------------------------------------
# SECTION 5: REPEATED-EVALUATION SETTINGS (IDENTICAL TO TASK 2)
# -----------------------------------------------------------------------------
# Every setting here is copied EXACTLY from the Task 2 raw script. Keeping
# these identical is what makes this a valid "improved vs. baseline"
# experiment -- the only thing that changed between the two scripts is the
# feature set built in Section 4 above.

N_SEEDS = 10                                                     # run the whole experiment 10 times, once per seed, as required
TRAIN_SUBSAMPLE_SIZE = 1000                                      # number of training rows actually used to fit eLCS in each run (same as Task 2)
TEST_EVAL_SIZE = 5000                                            # number of held-out test rows used to score each run (same as Task 2)
POPULATION_SIZE_N = round(1.5 * TRAIN_SUBSAMPLE_SIZE)             # eLCS population size, following the lecturer's "N ~ 1.5x training instances" guideline
LEARNING_ITERATIONS = 5 * TRAIN_SUBSAMPLE_SIZE                    # same iteration budget as Task 2, so the comparison isn't skewed by extra training time

print(f"Population size N = {POPULATION_SIZE_N}, learning_iterations = {LEARNING_ITERATIONS}")  # confirm the derived settings


# -----------------------------------------------------------------------------
# SECTION 6: RUN THE 10-SEED EVALUATION LOOP
# -----------------------------------------------------------------------------
# Because this script uses the SAME dataframe row order, the SAME y target,
# and the SAME seeds as Task 2's script, train_test_split will select the
# exact same rows for training and testing at every seed here as it did in
# Task 2 -- only the feature values attached to those rows differ (7 curated
# columns here vs. 40 raw columns there). This is what makes a fair, paired
# comparison possible in Task 5's statistical test.

per_seed_accuracy = []                                           # will hold one accuracy value per seed
all_true_labels = []                                              # will hold every true test label across all seeds, concatenated
all_predicted_labels = []                                         # will hold every predicted test label across all seeds, concatenated
last_model = None                                                 # keeps a reference to the final trained model, for exporting rules afterwards

for seed in range(N_SEEDS):                                       # loop once per random seed (0 through 9)
    X_train, X_test, y_train, y_test = train_test_split(            # split the FULL 150,000-row dataset for this seed
        X, y,                                                        # improved feature matrix and target vector
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

print(f"\nImproved eLCS -- mean accuracy over {N_SEEDS} seeds: {mean_accuracy:.4f} (std {std_accuracy:.4f})")  # headline result for Task 4
print("Compare this against Task 2's raw baseline mean accuracy to quantify the improvement.")  # reminder for the write-up


# -----------------------------------------------------------------------------
# SECTION 8: CONFUSION MATRIX AND PER-CLASS TP / TN / FP / FN
# -----------------------------------------------------------------------------
# Same pooling approach as Task 2: combine all 10 seeds' test predictions
# (10 x 5,000 = 50,000 evaluated rows) before computing the confusion matrix.

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
# We export the rule population from the LAST seed's trained model. These
# rules should be noticeably more compact and readable than the raw
# baseline's rules from Task 2, since they're built from only 7 meaningful
# columns instead of 40 mostly-noisy ones -- a good concrete point for the
# Task 7 interpretability discussion.

last_model.export_final_rule_population(
    headerNames=np.array(SELECTED_FEATURE_COLUMNS),                    # tell eLCS the real name of each of our 7 feature columns
    className="productivity_class",                                    # name of the target column in the exported file
    filename="task4_improved_lcs_rules.csv",                            # output filename
    DCAL=True,                                                          # export in the readable "Detailed Compact Attribute List" format
)

rule_population = pd.read_csv("task4_improved_lcs_rules.csv")           # read the exported rules back in
print(f"\nExported {len(rule_population)} rules to task4_improved_lcs_rules.csv")  # confirm how many rules were saved
print(rule_population.sort_values("Numerosity", ascending=False).head(5))         # preview the 5 most common rules


# -----------------------------------------------------------------------------
# SECTION 10: SAVE RESULTS FOR THE TASK 5/6 COMPARISON
# -----------------------------------------------------------------------------
# Same file structure as Task 2's script, so the two CSVs can be concatenated
# directly for the Task 5 statistical test and the Task 6 comparison table.

seed_results = pd.DataFrame({                                          # one row per seed, for later statistical comparison
    "seed": list(range(N_SEEDS)),
    "model": "Improved eLCS (7 selected features)",
    "accuracy": per_seed_accuracy,
})
seed_results.to_csv("task4_improved_lcs_seed_results.csv", index=False)  # save the per-seed results to their own CSV

summary_row = pd.DataFrame([{                                          # one summary row, for quick reference
    "model": "Improved eLCS (7 selected features)",
    "mean_accuracy": mean_accuracy,
    "std_accuracy": std_accuracy,
    "n_seeds": N_SEEDS,
    "n_features": X.shape[1],
    "n_rules_last_seed": len(rule_population),
}])
summary_row.to_csv("task4_improved_lcs_results.csv", index=False)       # save the summary row to its own CSV

print("\nSaved per-seed results to task4_improved_lcs_seed_results.csv and summary to task4_improved_lcs_results.csv")  # confirm the save
print(summary_row)                                                       # display the summary row in the output too
