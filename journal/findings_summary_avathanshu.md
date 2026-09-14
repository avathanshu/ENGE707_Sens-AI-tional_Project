Findings Summary — AI Adoption Dataset (Phase 1)

Dataset: ai_company_adoption.csv, 150,000 company-quarter records, 43 columns, 2023-2026.

Data quality: no missing values, no duplicate rows or response IDs, no logically inconsistent records (e.g. years_using_ai never exceeds company_age, no non-positive employee counts or revenue). The dataset is clean; Phase 1 cleaning mostly involved documenting this rather than fixing errors.

Adoption distribution: most company-quarters sit at pilot or partial adoption. None (no adoption) and full adoption are both rare, with full being the least common stage overall.

Investment and governance features (training hours, budget percentage, maturity score, failure rate) move together in a consistent way: they all increase from none to full adoption, except failure rate, which decreases as adoption stage rises. Companies further along the adoption journey report fewer AI project failures, not more.

ai_investment_per_employee is heavily right-skewed (skew about 4.3) and should be read alongside company size. Smaller companies show higher per-employee investment figures for the same budget, simply because that budget is spread across fewer staff. These high values were flagged, not removed, since there is no evidence they are errors.

Business outcomes (revenue growth, cost reduction, reskilled employees) all increase with adoption stage. Full-adoption companies show noticeably higher averages across all three than none/pilot companies.

Correlation check: training hours, budget percentage, and maturity score are positively correlated with each other and with the outcome variables. Failure rate is negatively correlated with all of these. ai_investment_per_employee has weaker correlations with the outcome variables than the other secondary features.

Overall takeaway: adoption maturity (not just whether a company has adopted AI at all) is associated with better business outcomes, and this association is consistent across training investment, budget allocation, and failure rate. This supports moving into Phase 2 with adoption stage as the prediction target and the secondary features as predictors.
