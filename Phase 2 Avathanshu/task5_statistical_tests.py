# =============================================================================
# TASK 5: PAIRED STATISTICAL TESTS ACROSS THE TEAM'S SHARED 10-SEED RESULTS
# -----------------------------------------------------------------------------
# Reads every *_seed_results.csv (seed, model, accuracy) produced by the shared
# scripts (task2, task3, task4, task6_*). Every model was scored on the SAME
# test rows for each seed (seeds 0-9, 80/20 stratified split, 5,000-row slice),
# so accuracies are PAIRED by seed. Tests used:
#   - paired t-test (assumes roughly normal differences)
#   - Wilcoxon signed-rank (non-parametric; with n=10 the smallest possible
#     two-sided p is ~0.002)
# Holm correction is applied across all pairwise comparisons.
# NOTE: only models whose seed CSV is present in this folder are compared.
# =============================================================================
import glob, itertools
import pandas as pd, numpy as np
from scipy import stats

frames = [pd.read_csv(f) for f in sorted(glob.glob("*seed_results.csv"))]
long = pd.concat(frames, ignore_index=True)
wide = long.pivot(index="seed", columns="model", values="accuracy")
print("Models found:", list(wide.columns), "| seeds:", len(wide))

summary = wide.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
print("\nMean accuracy over seeds:\n", summary.round(4))
summary.to_csv("task5_summary.csv")

rows = []
for a, b in itertools.combinations(wide.columns, 2):
    d = wide[a] - wide[b]
    t_p = stats.ttest_rel(wide[a], wide[b]).pvalue
    try:
        w_p = stats.wilcoxon(wide[a], wide[b]).pvalue
    except ValueError:
        w_p = np.nan
    rows.append(dict(model_a=a, model_b=b, mean_diff=d.mean(), ttest_p=t_p, wilcoxon_p=w_p))
res = pd.DataFrame(rows)

# Holm step-down correction on the Wilcoxon and t-test p-values
def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj
res["ttest_p_holm"] = holm(res["ttest_p"])
res["wilcoxon_p_holm"] = holm(res["wilcoxon_p"].fillna(1))
res["significant_holm_0.05"] = res["ttest_p_holm"] < 0.05
res.to_csv("task5_pairwise_tests.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print("\nPairwise paired tests:\n", res.round(5).to_string(index=False))
