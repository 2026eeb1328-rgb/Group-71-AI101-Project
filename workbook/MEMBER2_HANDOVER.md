# Member 2 — Preprocessing & Feature Selection Handover

## Dataset split
- Train: 1,400 rows (70%)
- Validation: 300 rows (15%)
- Test: 300 rows (15%)
- Stratified splitting; random states 42 and 43.
- All four classes remain balanced (350/75/75 observations per class in train/validation/test).

## Preprocessing decisions
- All 20 predictor columns are numeric.
- Median imputation for missing numeric values.
- StandardScaler for feature scaling.
- Imputation, Mutual Information selection and scaling are inside a single Scikit-learn pipeline, so every cross-validation fold fits transformations only on its own training fold.

## Feature selection experiment
- Method: SelectKBest(mutual_info_classif).
- Metric: macro F1.
- CV: 5-fold StratifiedKFold on training data only.
- Tested K: 5, 8, 10, 12, 15, 18, 20.
- Tested C: 0.01, 0.1, 1, 10, 100.
- Best observed configuration: K=18, C=100.
- Mean CV macro F1: 0.9550.
- Validation accuracy: 0.9767.
- Validation macro F1: 0.9767.

## Selected 18 features
battery_power, clock_speed, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi

Excluded by this experiment: `blue`, `dual_sim`.

## Important caveats
- These are observed results on the supplied CSV with the stated fixed split and search grid.
- K=18 beat K=20 by only about 0.0001 in mean CV macro F1. It is the best tested setting, not proof that 18 is universally optimal. A simpler or more stable choice can be discussed if the team prioritises parsimony.
- Validation metrics are not test metrics.
- Member 3 must not tune against the test set. Freeze the model configuration, then evaluate the test partition once.

## Scikit-learn compatibility
For current scikit-learn releases, `LogisticRegression(solver="lbfgs")` automatically uses multinomial loss for multiclass classification. The removed `multi_class` constructor argument is intentionally not passed.
