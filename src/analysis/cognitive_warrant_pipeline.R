sink("pipeline_log.txt", split = TRUE)
# ================================================================
# Cognitive Warrant Proof of Concept
# Restructuring and analysis pipeline for tb2023_exp2_lcsh_distances.csv
#
# Implements, on the real merged Experiment 2 / LCSH dataset, the pipeline
# described in Chapter 5 ("The Analytic Pipeline: Logistic Regression and
# the Estimation of Cognitive Warrant Metrics"):
#   Step 0  Restructure trial-level records to unique stimulus pairs
#   Step 0.5 Data diagnostics (run BEFORE trusting any downstream number)
#   Step 1  Fit baseline and full logistic models
#   Step 2  Convert fitted probabilities to conditional entropy
#   Step 3  Piecewise (binned) cognitive information, CI(d)
#   Step 4  Pointwise cognitive information, pCI_i
#   Step 5  Expected cognitive information (CI-bar) and warrant efficiency
# 
# Install required packages with: install.packages(c("tidyverse", "logistf", "lme4","splines","patchwork"))
# ================================================================

library(tidyverse)      # data analysis and visualization
library(logistf)        # fits a logistic regression model using Firth's bias reduction method
library(lme4)           # fits linear and generalized linear mixed-effects models
library(splines)        # provides functions for working with regression splines
library(patchwork)      # combines separate ggplots into the same graphic

set.seed(1226)

# ----------------------------------------------------------------
# Step 0: Load and restructure trial-level records to unique pairs
# ----------------------------------------------------------------

# Set working directory (comment this out to run)
setwd("C:/Users/nmc63/Projects/lcsh_polysemy_capstone/data/processed")
raw <- read_csv("tb2023_exp2_lcsh_distances.csv", show_col_types = FALSE)

# distance_elmo_lcsh is entirely missing in this file (LCSH-anchored ELMo
# frame was not computed) -- drop it rather than carry an all-NA column.
raw <- raw %>% select(-distance_elmo_lcsh)

# The trial-level file inherits the repeated-measures structure of
# Trott and Bergen's Experiment 2 (187 subjects, each responding to a subset
# of stimulus pairs). same, same_heading, hier_distance, distance_bert,
# ambiguity_type, and Class are all properties of the *stimulus pair*, not
# of the trial -- they take exactly one value within every (word,
# version_with_order) group. Confirmed empirically below; if this check
# ever fails for a future data pull, STOP and investigate before trusting
# the collapse.
pair_key <- c("word", "version_with_order")
pair_level_cols <- c("same", "same_heading", "hier_distance", "same_nbhd",
                      "distance_bert", "distance_bert_lcsh", "distance_elmo",
                      "ambiguity_type", "Class")

nun_check <- raw %>%
  group_by(across(all_of(pair_key))) %>%
  summarise(across(all_of(pair_level_cols), n_distinct), .groups = "drop")

stopifnot(all(nun_check[pair_level_cols] == 1))
message("Pair-level covariates confirmed constant within (word, version_with_order).")

# Trial-level summary (kept alongside the pair, in case RT/accuracy
# covariates are wanted for later robustness checks -- not used by the
# CI(d) pipeline itself, which operates on Y_i = `same`, not on RT/accuracy).
trial_summary <- raw %>%
  group_by(across(all_of(pair_key))) %>%
  summarise(
    n_trials       = n(),
    n_subjects     = n_distinct(subject),
    mean_correct   = mean(correct_response),
    mean_log_rt    = mean(log_rt),
    .groups = "drop"
  )

pairs_df <- raw %>%
  distinct(across(all_of(c(pair_key, pair_level_cols)))) %>%
  left_join(trial_summary, by = pair_key) %>%
  mutate(
    pair_id       = paste(word, version_with_order, sep = "__"),
    Y             = as.integer(same),           # Y_i: ground-truth same-sense indicator
    S             = as.integer(same_heading),   # S_i: LCSH heading congruence
    d             = distance_bert,              # d_i: BERT cosine distance (unmodified frame)
    h             = hier_distance               # h_i: BT/NT shortest-path distance (Inf if S_i = 0
                                                 #      with no shared ancestor within search horizon)
  )

message(sprintf(
  "Collapsed %d trial-level rows (%d subjects) into %d unique stimulus pairs across %d words.",
  nrow(raw), n_distinct(raw$subject), nrow(pairs_df), n_distinct(pairs_df$word)
))

# ----------------------------------------------------------------
# Step 1: Data diagnostics -- run before trusting any fitted model
# ----------------------------------------------------------------

# (a) Is S perfectly collinear with Y? This is the single most consequential
#     check in this pipeline. If TRUE, the "full model" logistic regression
#     of Y on d and S is not testing whether LCSH heading congruence adds
#     information beyond BERT distance -- it is recovering a tautology,
#     because S was constructed to equal Y for every pair in this file
#     (each dictionary sense M1/M2 was mapped one-to-one onto a single
#     LCSH authority record). CI(d) will come out equal to H(Y|d) exactly
#     (up to numerical floor), and warrant efficiency eta(d) will be ~1
#     everywhere. That is a mechanical consequence of how same_heading was
#     derived here, not an empirical finding about LCSH's cognitive value.
same_vs_heading <- mean(pairs_df$Y == pairs_df$S)
message(sprintf("Proportion of pairs where Y (same) == S (same_heading): %.4f", same_vs_heading))
if (isTRUE(all.equal(same_vs_heading, 1))) {
  warning(paste(
    "same_heading is IDENTICAL to same for every pair in this file.",
    "The baseline-vs-full logistic model comparison below will show a",
    "trivial, perfectly-separated effect of S -- this reflects how the",
    "LCSH mapping was constructed (one heading per dictionary sense), not",
    "an empirical test of whether LCSH boundaries carry information beyond",
    "distance. Consider whether hier_distance (h), which DOES vary",
    "independently of Y among cross-heading pairs, is the more informative",
    "structural variable to foreground in this proof of concept -- see the",
    "'Structural extension' block near the end of this script."
  ))
}

# (b) hier_distance is Inf for cross-heading pairs with no shared ancestor
#     within the search horizon, and exactly 0 for same-heading pairs
#     (trivially, since same node has zero path length to itself). It is
#     only *non-trivially* informative within the cross-heading (S = 0)
#     subset, where it takes finite values > 0 for pairs that share a
#     broader term and Inf for pairs that do not.
message("hier_distance cross-tab (S = same_heading x finite/inf/zero):")
print(pairs_df %>%
  mutate(h_cat = case_when(
    is.infinite(h) ~ "inf",
    h == 0         ~ "zero",
    TRUE           ~ "finite>0"
  )) %>%
  count(S, h_cat))

# (c) distance_bert range and coverage of the theoretically critical
#     intermediate zone (~0.3-0.6 cosine distance as
#     the provisional region where CI(d) should peak).
message(sprintf("distance_bert range: [%.3f, %.3f]", min(pairs_df$d), max(pairs_df$d)))
message(sprintf(
  "Proportion of pairs with d in [0.30, 0.60]: %.3f",
  mean(pairs_df$d >= 0.30 & pairs_df$d <= 0.60)
))

# ----------------------------------------------------------------
# Step 2: Fit baseline and full logistic models
# ----------------------------------------------------------------

# Baseline: logit P(Y = 1 | d) = beta0 + beta1 * d
m_baseline <- glm(Y ~ d, data = pairs_df, family = binomial(link = "logit"))

# Full: logit P(Y = 1 | d, S) = gamma0 + gamma1 * d + gamma2 * S
# NOTE: given the Step 0.5(a) diagnostic, this model is expected to show
# quasi/perfect separation on S (glm will warn "fitted probabilities
# numerically 0 or 1 occurred", and the SE on gamma2 will be enormous).
# The point estimates and fitted probabilities are still usable for the
# entropy calculations below, but reported p-values / CIs for gamma2 are
# not trustworthy from glm alone in this degenerate case.
m_full_glm <- glm(Y ~ d + S, data = pairs_df, family = binomial(link = "logit"))

if (!m_full_glm$converged || any(abs(coef(m_full_glm)) > 15)) {
  message("m_full_glm shows signs of (quasi-)separation, as expected -- fitting Firth's bias-reduced logistic regression as a stable alternative.")
}

# Firth (bias-reduced) logistic regression: does not blow up under perfect
# separation, and yields finite, interpretable fitted probabilities even
# when S is a near-deterministic function of Y. Preferred for the entropy
# calculations in Steps 2-5 whenever m_full_glm shows separation.
m_full_firth <- logistf(Y ~ d + S, data = pairs_df)

summary(m_baseline)
summary(m_full_glm)
summary(m_full_firth)

# Likelihood-ratio test: the regression-side estimate of CI(d) > 0, i.e.,
# the finite-sample test of I(Y; S | d) = 0. Reported for completeness;
# interpret cautiously given the separation diagnosed above.
anova(m_baseline, m_full_glm, test = "Chisq")

# Optional robustness check: account for the fact that pairs are nested
# within word (25 words, ~12 pairs each) via a random intercept for word.
# Likely to be singular/unstable for the same separation reason, but worth
# inspecting.
m_full_glmer <- tryCatch(
  glmer(Y ~ d + S + (1 | word), data = pairs_df, family = binomial),
  warning = function(w) { message("glmer warning (expected under separation): ", conditionMessage(w)); NULL },
  error   = function(e) { message("glmer error: ", conditionMessage(e)); NULL }
)

# Fitted probabilities. Use the Firth model for the full-model probability
# as the primary specification; glm's are retained for comparison.
pairs_df <- pairs_df %>%
  mutate(
    p_baseline      = predict(m_baseline, type = "response"),
    p_full_glm      = predict(m_full_glm, type = "response"),
    p_full          = plogis(predict(m_full_firth, newdata = pairs_df, type = "link"))
  )

# ----------------------------------------------------------------
# Step 3: From fitted probabilities to conditional entropy
# ----------------------------------------------------------------

h_b <- function(p) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)  # guard against log(0)
  -p * log2(p) - (1 - p) * log2(1 - p)
}

pairs_df <- pairs_df %>%
  mutate(
    H_baseline = h_b(p_baseline),
    H_full     = h_b(p_full)
  )

# ----------------------------------------------------------------
# Step 4: Piecewise (binned) cognitive information, CI(d)
# ----------------------------------------------------------------

K <- 6  # fewer bins than in the schematic Ch. 5 template, given n = 299
         # pairs; re-run with K = 8, 10 as a sensitivity check.
pairs_df <- pairs_df %>%
  mutate(distance_bin = cut(d, breaks = K, include.lowest = TRUE))

ci_piecewise <- pairs_df %>%
  group_by(distance_bin) %>%
  summarise(
    n_pairs  = n(),
    x_mid    = mean(d),
    H_bar_d  = mean(H_baseline),
    H_bar_dS = mean(H_full),
    CI_k     = H_bar_d - H_bar_dS,
    # CORRECTED (see H_true_S below for the full derivation): H_bar_d above
    # comes from m_baseline, a model fit independently of m_full_firth, so
    # CI_k is not guaranteed non-negative by Jensen's inequality -- that
    # guarantee requires H(Y|d) to be the entropy of the SAME model's own
    # S-marginalized mixture probability. Within a bin, every pair has an
    # actual observed S, so no local-window weight estimate is even needed
    # here (unlike the grid version): mean(p_full) IS the bin's true,
    # empirical S-marginalized probability. Exposed as its own column
    # (not just its entropy) so Step 4 can reuse it as pCI's marginal
    # reference, replacing p_baseline -- see Step 4 for why.
    p_bar_full  = mean(p_full),
    H_bar_d_mix = h_b(p_bar_full),
    CI_k_v2     = H_bar_d_mix - H_bar_dS,
    P_d      = n_pairs / nrow(pairs_df),
    .groups = "drop"
  )

print(ci_piecewise)

# Smooth, model-implied CI(d) curve on a fine grid, for plotting against
# the piecewise estimate above and against the theoretical inverted-U
# prediction (Chapter 4, H3-as-theorem).
grid_d <- tibble(d = seq(min(pairs_df$d), max(pairs_df$d), length.out = 200))
grid_d <- grid_d %>%
  mutate(
    p_baseline_grid = predict(m_baseline, newdata = grid_d, type = "response"),
    p_full_grid_S0  = plogis(predict(m_full_firth, newdata = mutate(grid_d, S = 0), type = "link")),
    p_full_grid_S1  = plogis(predict(m_full_firth, newdata = mutate(grid_d, S = 1), type = "link")),
    H_baseline_grid = h_b(p_baseline_grid)
  )
# Because S is (quasi-)deterministic given d in this dataset, the
# population-averaged H(Y | d, S) on the grid should be computed by
# weighting S = 0 / S = 1 curves by the empirical P(S = 1 | d) at each d,
# not by assuming a fixed S. A local (binned) empirical weight is used
# here as a simple approximation:
grid_d <- grid_d %>%
  rowwise() %>%
  mutate(
    pi_S1 = mean(pairs_df$S[abs(pairs_df$d - d) <= 0.05]),
    pi_S1 = ifelse(is.nan(pi_S1), mean(pairs_df$S), pi_S1),
    H_full_grid = pi_S1 * h_b(p_full_grid_S1) + (1 - pi_S1) * h_b(p_full_grid_S0),
    CI_grid = H_baseline_grid - H_full_grid
  ) %>%
  ungroup()

# CORRECTED: CI_grid above uses H_baseline_grid from m_baseline, a model
# fit independently of m_full_firth -- Jensen's inequality (the source of
# CI(d)'s non-negativity) only guarantees H(Y|d) - E_S[H(Y|S,d)] >= 0 when
# H(Y|d) is the entropy of that SAME model's own S-marginalized mixture
# probability, not an independently-estimated stand-in. This mirrors the
# fix validated for the hier_distance model below (0/85 negative grid
# points at d > 0.4 once corrected, vs. 85/85 before).
grid_d <- grid_d %>%
  mutate(
    p_mix_S  = pi_S1 * p_full_grid_S1 + (1 - pi_S1) * p_full_grid_S0,
    H_true_S = h_b(p_mix_S),
    CI_grid_v2 = H_true_S - H_full_grid
  )

p_ci_S <- ggplot(grid_d, aes(d, CI_grid_v2)) + geom_line() +
   geom_point(data = ci_piecewise, aes(x_mid, CI_k_v2), inherit.aes = FALSE) +
   labs(x = "BERT cosine distance (d)", y = "CI(d), bits", title = "S (same_heading)")
print(p_ci_S)

# ----------------------------------------------------------------
# Step 5: Pointwise cognitive information, pCI_i
# ----------------------------------------------------------------

pairs_df <- pairs_df %>%
  mutate(
    p_full_obs     = ifelse(Y == 1, p_full,     1 - p_full),
    p_baseline_obs = ifelse(Y == 1, p_baseline, 1 - p_baseline),
    pCI            = log2(p_full_obs / p_baseline_obs)
  )

# CORRECTED: pCI above uses p_baseline (from m_baseline, fit independently
# of m_full_firth) as its marginal reference. The "by construction" identity
# E[pCI_i | bin k] = CI_k is a claim about an in-EXPECTATION relationship
# between two log-probability terms computed from the SAME pair of
# reference models; it only holds if pCI and CI_k are built from consistent
# models. CI_k_v2 was corrected to use p_bar_full (the full model's own
# bin-mean, S-marginalized probability, from ci_piecewise) instead of
# m_baseline -- so for pCI to remain internally consistent with CI_k_v2,
# its marginal reference needs the same substitution. Using p_baseline here
# instead introduces a residual term equal to the KL divergence between the
# true (mixture) distribution and m_baseline's fitted distribution within
# each pair's bin -- not sampling noise, but a genuine, bin-varying bias
# that grows wherever m_baseline is a poor local approximation (exactly the
# sparse, high-distance bins where the pre-correction consistency check
# showed its largest gaps).
pairs_df <- pairs_df %>%
  left_join(ci_piecewise %>% select(distance_bin, p_bar_full), by = "distance_bin") %>%
  mutate(
    p_marginal_obs = ifelse(Y == 1, p_bar_full, 1 - p_bar_full),
    pCI_v2         = log2(p_full_obs / p_marginal_obs)
  )

# Internal consistency check: mean(pCI_v2) within a bin should approximate
# that bin's piecewise CI_k_v2 (they are computed from different formulas
# at different levels of aggregation; agreement indicates the pipeline
# implementation, not just the theory, is behaving as specified). Uses
# pCI_v2 (corrected), not pCI -- see the note above.
consistency_check <- pairs_df %>%
  group_by(distance_bin) %>%
  summarise(mean_pCI_v2 = mean(pCI_v2), .groups = "drop") %>%
  left_join(ci_piecewise, by = "distance_bin") %>%
  mutate(diff = mean_pCI_v2 - CI_k_v2)
print(consistency_check)

# Heading-reform diagnostic preview: pairs with the most negative pCI_v2 are
# candidates where the LCSH heading assignment, retroactively applied to
# Trott & Bergen's stimuli, misdirects a purely distributional judgment
# about sense identity. Given the Step 0.5(a) finding, in THIS dataset a
# negative pCI_v2 can only arise from a genuine bin-marginal-vs-full model
# disagreement near the fitted decision boundary, not from S disagreeing
# with Y (since S = Y here) -- interpret accordingly. pCI (uncorrected) is
# retained alongside for comparison against the previously reported ranking.
pairs_df %>%
  select(pair_id, word, ambiguity_type, Y, S, h, d, pCI, pCI_v2) %>%
  arrange(pCI_v2) %>%
  slice_head(n = 10) %>%
  print()

# ----------------------------------------------------------------
# Step 6: Expected cognitive information and warrant efficiency
# ----------------------------------------------------------------

# Uses CI_k_v2 (corrected) -- see the H_bar_d_mix fix in ci_piecewise above.
CI_bar  <- with(ci_piecewise, sum(P_d * CI_k_v2))
eta_bar <- CI_bar / with(ci_piecewise, sum(P_d * H_bar_d_mix))

message(sprintf("Expected cognitive information, CI-bar = %.4f bits", CI_bar))
message(sprintf("Warrant efficiency, eta-bar = %.4f", eta_bar))

# ----------------------------------------------------------------
# Structural extension: hier_distance as the non-tautological predictor
# ----------------------------------------------------------------
# Because S = Y exactly in this file (Step 0.5a), the S-based CI(d) above
# is a ceiling artifact of the mapping, not an empirical result. hier_distance
# (h) is NOT tautologically fixed by Y: among cross-heading pairs (S = 0),
# h takes finite values for pairs sharing a broader term and Inf otherwise,
# and this variation is independent of the same/different-sense label by
# construction. A model of Y using d and a finite/inf recoding of h --
# rather than S -- is arguably the more informative proof-of-concept
# computation on this particular file, since it is not guaranteed to
# succeed by the way the variables were built.

pairs_df <- pairs_df %>%
  mutate(
    h_finite = is.finite(h),                       # 1 if a shared BT/NT ancestor exists within horizon
    h_capped = ifelse(is.finite(h), h, NA_real_)    # finite hierarchical distances only
  )

m_full_hier <- glm(Y ~ d + h_finite, data = pairs_df, family = binomial(link = "logit"))
summary(m_full_hier)
anova(m_baseline, m_full_hier, test = "Chisq")

# CORRECTION (relative to the original comment here): m_full_hier is NOT a
# clean, non-separated fit. h_finite = FALSE implies Y = 0 for all 119 such
# pairs with zero exceptions -- a single categorical level with a 0% event
# rate is sufficient to induce quasi-complete separation, independent of
# whether the other level (h_finite = TRUE) mixes both classes. This is
# visible directly in summary(m_full_hier): the coefficient on
# h_finiteTRUE (19.29) has an enormous SE (936.97), giving z = 0.02,
# p = 0.98 -- the classic large-estimate/large-SE/near-zero-z signature of
# separation. That Wald p-value is not usable. The LRT is: anova() above
# gives deviance = 89.43 on 1 df, p < 2.2e-16, which (unlike the Wald test)
# remains valid under separation because it compares log-likelihoods
# directly rather than relying on the asymptotic normality of the
# coefficient estimate. Report the LRT, not summary()'s p-value, for
# h_finite's effect -- and note this is a materially different situation
# from the S model, where the LRT reflects the residual deviance collapsing
# essentially to 0 (a perfect fit), not merely a strong effect.
#
# For the probability/entropy pipeline below, refit via Firth's
# bias-reduced logistic regression, exactly as for the S model, so that
# fitted probabilities (and everything downstream) come from a stable,
# finite estimate rather than the quasi-separated glm.
m_full_hier_firth <- logistf(Y ~ d + h_finite, data = pairs_df)
summary(m_full_hier_firth)

# ----------------------------------------------------------------
# Steps 3-6, re-run for m_full_hier (using the Firth refit)
# ----------------------------------------------------------------

# Step 3 (hier): fitted probability -> conditional entropy
pairs_df <- pairs_df %>%
  mutate(
    p_full_hier = plogis(predict(m_full_hier_firth, newdata = pairs_df, type = "link")),
    H_full_hier = h_b(p_full_hier)
  )

# Step 4 (hier): piecewise cognitive information
# Reuses the same `distance_bin` factor created for the S-based table, so
# the two piecewise tables are directly comparable bin-for-bin.
ci_piecewise_hier <- pairs_df %>%
  group_by(distance_bin) %>%
  summarise(
    n_pairs   = n(),
    x_mid     = mean(d),
    H_bar_d   = mean(H_baseline),
    H_bar_dh  = mean(H_full_hier),
    CI_k_hier = H_bar_d - H_bar_dh,
    # CORRECTED, same reasoning as ci_piecewise's H_bar_d_mix/CI_k_v2 above:
    # H_bar_d uses m_baseline, fit independently of m_full_hier_firth, so
    # non-negativity isn't guaranteed. mean(p_full_hier) is the bin's true
    # empirical h_finite-marginalized probability (each pair's actual
    # observed h_finite already went into computing p_full_hier), so no
    # local-window weight estimate is needed at the piecewise level.
    # Exposed as its own column (not just its entropy) for the same reason
    # as ci_piecewise's p_bar_full -- Step 4 (hier) reuses it as pCI_hier's
    # marginal reference.
    p_bar_full_hier  = mean(p_full_hier),
    H_bar_d_mix_hier = h_b(p_bar_full_hier),
    CI_k_hier_v2     = H_bar_d_mix_hier - H_bar_dh,
    P_d       = n_pairs / nrow(pairs_df),
    .groups = "drop"
  )
print(ci_piecewise_hier)

# Smooth grid curve: replace the S = 0/1 reweighting block with the
# h_finite = FALSE/TRUE analogue, predicting from m_full_hier_firth (not
# the quasi-separated m_full_hier) -- link-scale predict + plogis(), same
# pattern as the S-based grid above.
grid_d <- grid_d %>%
  mutate(
    p_full_grid_h0 = plogis(predict(m_full_hier_firth, newdata = mutate(grid_d, h_finite = FALSE), type = "link")),
    p_full_grid_h1 = plogis(predict(m_full_hier_firth, newdata = mutate(grid_d, h_finite = TRUE),  type = "link"))
  ) %>%
  rowwise() %>%
  mutate(
    pi_h1 = mean(pairs_df$h_finite[abs(pairs_df$d - d) <= 0.05]),
    pi_h1 = ifelse(is.nan(pi_h1), mean(pairs_df$h_finite), pi_h1),
    H_full_grid_hier = pi_h1 * h_b(p_full_grid_h1) + (1 - pi_h1) * h_b(p_full_grid_h0),
    CI_grid_hier = H_baseline_grid - H_full_grid_hier
  ) %>%
  ungroup()

# Diagnostic: is negative CI_grid_hier (if any) explained by empty local
# h_finite = TRUE support, forcing pi_h1's NaN fallback to the *global*
# mean(h_finite) rather than a locally-appropriate value? Check wherever
# CI_grid_hier is negative -- if n_local_h1 is 0 (or n_local_total is 0,
# triggering the global fallback) throughout, that confirms the mechanism.
grid_d %>%
  filter(CI_grid_hier < 0) %>%
  rowwise() %>%
  mutate(n_local_h1    = sum(pairs_df$h_finite[abs(pairs_df$d - d) <= 0.05] == TRUE),
         n_local_total = sum(abs(pairs_df$d - d) <= 0.05)) %>%
  ungroup() %>%
  select(d, pi_h1, n_local_h1, n_local_total, CI_grid_hier) %>%
  print(n = Inf)

# Diagnostic (revised hypothesis): the local-support story above is ruled
# out by the check above (pi_h1 is a genuine local estimate, not a global
# fallback, throughout the negative-CI region -- it even reaches 1.0). The
# alternative hypothesis: h_finite = TRUE is a mixture of two different
# populations -- same-heading pairs (h = 0, trivially finite, Y = 1 by
# construction, clustering at low d) and cross-heading pairs that merely
# share *some* ancestor within the search horizon (h > 0, Y unconstrained).
# If that mixture's composition flips across d -- mostly same-heading/Y=1
# at low d, mostly cross-heading/Y=0 at high d -- then m_full_hier's
# additive (no d:h_finite interaction) specification can't represent it,
# and the coefficient (fit mostly against the low-d same-heading mass)
# gets extrapolated, unchanged, into a high-d regime it doesn't describe.
pairs_df %>%
  mutate(d_region = ifelse(d > 0.4, "d > 0.4", "d <= 0.4")) %>%
  filter(h_finite) %>%
  group_by(d_region) %>%
  summarise(n = n(), prop_same_heading = mean(S == 1), mean_Y = mean(Y), .groups = "drop")

# Diagnostic (interaction check): the composition check above confirms
# h_finite = TRUE flips from mostly same-heading/Y=1 at low d to mostly
# cross-heading/Y=0 at high d. m_full_hier's additive specification can't
# represent that -- test whether letting h_finite's effect vary with d
# (d * h_finite) shrinks or removes the negative-CI region above d = 0.4.
# Reuses pi_h1 already computed on grid_d (the local-weighting scheme
# itself isn't in question here, only the model's functional form).
m_full_hier_int <- logistf(Y ~ d * h_finite, data = pairs_df)
summary(m_full_hier_int)

grid_d_int <- grid_d %>%
  mutate(
    p_h0_int   = plogis(predict(m_full_hier_int, newdata = mutate(grid_d, h_finite = FALSE), type = "link")),
    p_h1_int   = plogis(predict(m_full_hier_int, newdata = mutate(grid_d, h_finite = TRUE),  type = "link")),
    H_full_int = pi_h1 * h_b(p_h1_int) + (1 - pi_h1) * h_b(p_h0_int),
    CI_grid_hier_int = H_baseline_grid - H_full_int
  )

message(sprintf(
  "Additive model:    %d/%d grid points with d > 0.4 have CI_grid_hier < 0 (min = %.4f)",
  sum(grid_d$CI_grid_hier[grid_d$d > 0.4] < 0, na.rm = TRUE),
  sum(grid_d$d > 0.4),
  min(grid_d$CI_grid_hier[grid_d$d > 0.4], na.rm = TRUE)
))
message(sprintf(
  "Interaction model: %d/%d grid points with d > 0.4 have CI_grid_hier < 0 (min = %.4f)",
  sum(grid_d_int$CI_grid_hier_int[grid_d_int$d > 0.4] < 0, na.rm = TRUE),
  sum(grid_d_int$d > 0.4),
  min(grid_d_int$CI_grid_hier_int[grid_d_int$d > 0.4], na.rm = TRUE)
))

m_baseline_firth <- logistf(Y ~ d, data = pairs_df)

grid_d_firthbase <- grid_d %>%
  mutate(
    p_baseline_firth = plogis(predict(m_baseline_firth, newdata = grid_d, type = "link")),
    H_baseline_firth  = h_b(p_baseline_firth),
    CI_grid_hier_firthbase = H_baseline_firth - H_full_grid_hier
  )

message(sprintf(
  "Firth/Firth comparison: %d/%d grid points with d > 0.4 have CI < 0 (min = %.4f)",
  sum(grid_d_firthbase$CI_grid_hier_firthbase[grid_d_firthbase$d > 0.4] < 0, na.rm = TRUE),
  sum(grid_d_firthbase$d > 0.4),
  min(grid_d_firthbase$CI_grid_hier_firthbase[grid_d_firthbase$d > 0.4], na.rm = TRUE)
))


# CORRECTED (root cause, confirmed by ruling out the two hypotheses above):
# CI_grid_hier uses H_baseline_grid from m_baseline, a model fit
# independently of m_full_hier_firth -- Jensen's inequality only guarantees
# non-negativity when H(Y|d) is the entropy of that SAME model's own
# h_finite-marginalized mixture probability, not an independently-estimated
# stand-in. Neither the model's functional form (additive vs. interaction,
# tested above) nor the estimation method (unpenalized vs. Firth, tested
# above) was the driver -- it's this structural mismatch. Validated: 0/85
# grid points negative at d > 0.4 once corrected (vs. 85/85 before).
grid_d <- grid_d %>%
  mutate(
    p_mix_hier = pi_h1 * p_full_grid_h1 + (1 - pi_h1) * p_full_grid_h0,
    H_true_hier = h_b(p_mix_hier),
    CI_grid_hier_v2 = H_true_hier - H_full_grid_hier
  )

# ggplot for the structural extension: CI(d) using h_finite (hier_distance)
# in place of S, as a separate plot from the S-based one above -- same
# structure (smooth model-implied curve + piecewise binned points), reading
# from grid_d$CI_grid_hier_v2 / ci_piecewise_hier$CI_k_hier_v2 (corrected)
# instead of grid_d$CI_grid_v2 / ci_piecewise$CI_k_v2, so the two are
# directly comparable side by side.
p_ci_hier <- ggplot(grid_d, aes(d, CI_grid_hier_v2)) + geom_line() +
   geom_point(data = ci_piecewise_hier, aes(x_mid, CI_k_hier_v2), inherit.aes = FALSE) +
   labs(x = "BERT cosine distance (d)", y = "CI(d), bits",
        title = "hier_distance (h_finite)")
print(p_ci_hier)

# Side-by-side comparison of the S-based and hier_distance-based CI(d)
# estimates (p_ci_S, p_ci_hier above). Both panels are put on the same
# y-axis range -- otherwise each would auto-scale to its own data and a
# visual comparison of *magnitude* between the two would be misleading,
# even though the shapes would still be individually correct.
ci_vals <- c(grid_d$CI_grid_v2, grid_d$CI_grid_hier_v2,
             ci_piecewise$CI_k_v2, ci_piecewise_hier$CI_k_hier_v2)
shared_ylim <- range(ci_vals[is.finite(ci_vals)])

p_ci_compare <-
  (p_ci_S    + coord_cartesian(ylim = shared_ylim)) |
  (p_ci_hier + coord_cartesian(ylim = shared_ylim))
p_ci_compare <- p_ci_compare +
  plot_annotation(title = "CI(d): S (same_heading) vs. hier_distance (h_finite)")
print(p_ci_compare)

# Overlay version: both series on one set of axes, distinguished by color,
# rather than two side-by-side panels -- lets you read off both CI(d)
# values at the same d directly, instead of eyeballing across two plots.
grid_compare <- bind_rows(
  grid_d %>% transmute(d, CI = CI_grid_v2,      model = "S (same_heading)"),
  grid_d %>% transmute(d, CI = CI_grid_hier_v2, model = "hier_distance (h_finite)")
)
piecewise_compare <- bind_rows(
  ci_piecewise      %>% transmute(x_mid, CI = CI_k_v2,      model = "S (same_heading)"),
  ci_piecewise_hier %>% transmute(x_mid, CI = CI_k_hier_v2, model = "hier_distance (h_finite)")
)

# Two legend dimensions: `color` distinguishes the model (S / h_finite),
# and `linetype`/`shape` distinguish the smooth model-implied curve from the
# piecewise binned estimates -- each mapped to a single descriptive string
# (there's only one line style and one point shape in use), which is the
# standard ggplot idiom for adding a labeled legend key for a constant
# aesthetic. Point size is bumped up (default is 1.5) so the piecewise
# estimates aren't visually dominated by the line.
p_ci_overlay <- ggplot(grid_compare, aes(d, CI, color = model)) +
  geom_line(aes(linetype = "Smooth curve (model-implied)")) +
  geom_point(data = piecewise_compare,
             aes(x_mid, CI, color = model, shape = "Piecewise estimate (binned)"),
             inherit.aes = FALSE, size = 3.5) +
  labs(x = "BERT cosine distance (d)", y = "CI(d), bits",
       color = "Predictor", linetype = NULL, shape = NULL,
       title = "CI(d): S (same_heading) vs. hier_distance (h_finite)") +
  guides(shape = guide_legend(override.aes = list(size = 3.5)))
print(p_ci_overlay)

# Step 4 (hier): pointwise cognitive information
pairs_df <- pairs_df %>%
  mutate(
    p_full_hier_obs = ifelse(Y == 1, p_full_hier, 1 - p_full_hier),
    pCI_hier        = log2(p_full_hier_obs / p_baseline_obs)   # p_baseline_obs unchanged from the S-based block
  )

# CORRECTED, same reasoning as the S-based pCI_v2 fix above: p_baseline is
# replaced with p_bar_full_hier (from ci_piecewise_hier), the h_finite-
# marginalized full model's own bin-mean probability, so pCI_hier_v2 stays
# internally consistent with CI_k_hier_v2 rather than carrying a residual
# KL-divergence bias from m_baseline's local misspecification.
pairs_df <- pairs_df %>%
  left_join(ci_piecewise_hier %>% select(distance_bin, p_bar_full_hier), by = "distance_bin") %>%
  mutate(
    p_marginal_hier_obs = ifelse(Y == 1, p_bar_full_hier, 1 - p_bar_full_hier),
    pCI_hier_v2          = log2(p_full_hier_obs / p_marginal_hier_obs)
  )

# Compared against CI_k_hier_v2 (corrected), same reasoning as the S-based
# consistency_check above. Uses pCI_hier_v2, not pCI_hier.
consistency_check_hier <- pairs_df %>%
  group_by(distance_bin) %>%
  summarise(mean_pCI_hier_v2 = mean(pCI_hier_v2), .groups = "drop") %>%
  left_join(ci_piecewise_hier, by = "distance_bin") %>%
  mutate(diff = mean_pCI_hier_v2 - CI_k_hier_v2)
print(consistency_check_hier)

# Ranking now uses h/h_finite rather than S -- this is the genuine
# heading-reform-style diagnostic for this file, since h_finite is not
# definitionally tied to Y the way S is. pCI_hier (uncorrected) is retained
# alongside for comparison against the previously reported ranking.
pairs_df %>%
  select(pair_id, word, ambiguity_type, Y, S, h, h_finite, d, pCI_hier, pCI_hier_v2) %>%
  arrange(pCI_hier_v2) %>%
  slice_head(n = 10) %>%
  print()

pairs_df %>%
  select(pair_id, word, ambiguity_type, h, d, pCI_hier, pCI_hier_v2) %>%
  arrange(pCI_hier_v2) %>%
  slice_head(n = 10) %>%
  as.data.frame()   # avoids tibble's column-count truncation

# Step 6 (hier): expected cognitive information and warrant efficiency
# Uses CI_k_hier_v2 / H_bar_d_mix_hier (corrected) -- see the fix in
# ci_piecewise_hier above.
CI_bar_hier  <- with(ci_piecewise_hier, sum(P_d * CI_k_hier_v2))
eta_bar_hier <- CI_bar_hier / with(ci_piecewise_hier, sum(P_d * H_bar_d_mix_hier))

message(sprintf("Expected cognitive information (hier_distance model), CI-bar = %.4f bits", CI_bar_hier))
message(sprintf("Warrant efficiency (hier_distance model), eta-bar = %.4f", eta_bar_hier))

# S-based peak (corrected: CI_grid_v2, not CI_grid)
grid_d$d[which.max(grid_d$CI_grid_v2)]        # -> D_S
max(grid_d$CI_grid_v2)                        # -> X_S

# hier_distance-based peak (corrected: CI_grid_hier_v2, not CI_grid_hier)
grid_d$d[which.max(grid_d$CI_grid_hier_v2)]   # -> D_H
max(grid_d$CI_grid_hier_v2)                   # -> X_H


neg_v2 <- pairs_df$pCI_hier_v2 < 0
hy0    <- pairs_df$h_finite & (pairs_df$Y == 0)
sum(neg_v2)                 # how many pairs are negative under the corrected version
sum(hy0)                    # should still be 80, unaffected by the pCI fix
sum(neg_v2 != hy0)          # 0 if the two sets still match exactly


# The 28 pairs newly flagged as negative under pCI_hier_v2, that weren't
# part of the old (h_finite=TRUE & Y=0) characterization:
new_negatives <- pairs_df %>%
  filter(pCI_hier_v2 < 0, !(h_finite & Y == 0))
table(h_finite = new_negatives$h_finite, Y = new_negatives$Y)

# The 10 pairs that WERE in the old group but are no longer flagged:
dropped <- pairs_df %>%
  filter(!(pCI_hier_v2 < 0), h_finite, Y == 0)
nrow(dropped)
dropped %>% select(pair_id, word, h, d, pCI_hier, pCI_hier_v2) %>% as.data.frame()

# And the 10 most negative among the newly-entered 28, so I know which
# specific pairs (if any) belong in the manuscript's named examples:
new_negatives %>%
  arrange(pCI_hier_v2) %>%
  slice_head(n = 10) %>%
  select(pair_id, word, h_finite, Y, h, d, pCI_hier_v2) %>%
  as.data.frame()


# ----------------------------------------------------------------
# Step 7: Saving output
# ----------------------------------------------------------------
# Two different things are saved here: (a) the analysis-ready tables and
# fitted model objects, so downstream work (plots, the dissertation
# write-up, re-checking a number six months from now) doesn't require
# re-running the whole pipeline -- re-fitting m_full_firth via logistf is
# the slow step; and (b) a full console transcript, for an audit trail of
# exactly what a given run printed (fitted coefficients, LRTs, warnings).
#
# For (a full console transcript), the simplest approach is external to
# this script: run it with
#   Rscript cognitive_warrant_pipeline.R > pipeline_log.txt 2>&1
# which captures print()/summary()/message()/warning() output exactly as
# it appeared. If running interactively instead, bracket the whole script
# with sink("pipeline_log.txt", split = TRUE) at the top and sink() at
# the bottom (split = TRUE echoes to the console as well as the file).

out_dir <- "pipeline_output"
dir.create(out_dir, showWarnings = FALSE)

# --- Plots, as PNG (for dropping into the write-up) ---
ggsave(file.path(out_dir, "p_ci_overlay.png"), plot = p_ci_overlay,
       width = 7, height = 5, units = "in", dpi = 300)
ggsave(file.path(out_dir, "p_ci_hier.png"), plot = p_ci_hier,
       width = 7, height = 5, units = "in", dpi = 300)

# --- Tables, as CSV (for re-use, e.g. in a separate plotting script) ---
write_csv(pairs_df,               file.path(out_dir, "pairs_df_full.csv"))
write_csv(ci_piecewise,           file.path(out_dir, "ci_piecewise_S.csv"))
write_csv(ci_piecewise_hier,      file.path(out_dir, "ci_piecewise_hier.csv"))
write_csv(consistency_check,      file.path(out_dir, "consistency_check_S.csv"))
write_csv(consistency_check_hier, file.path(out_dir, "consistency_check_hier.csv"))

# --- Fitted model objects, as .rds (cheap to reload; avoids re-fitting) ---
saveRDS(m_baseline,        file.path(out_dir, "m_baseline.rds"))
saveRDS(m_full_glm,        file.path(out_dir, "m_full_glm.rds"))         # quasi-separated glm, S -- kept for the record
saveRDS(m_full_firth,      file.path(out_dir, "m_full_firth.rds"))       # Firth refit, S -- used for p_full
saveRDS(m_full_hier,       file.path(out_dir, "m_full_hier.rds"))        # quasi-separated glm, h_finite -- LRT only
saveRDS(m_full_hier_firth, file.path(out_dir, "m_full_hier_firth.rds"))  # Firth refit, h_finite -- used for p_full_hier
# Reload later with e.g. m_full_firth <- readRDS("pipeline_output/m_full_firth.rds")

# --- LaTeX-ready table code, for dropping straight into Chapter 5 ---
# Populates the Table~\ref{tab:ci-bar-schematic} placeholder from the
# Ch. 5 .tex file. install.packages("knitr") if not already available.
# Uses CI_k_v2 (corrected) -- see the H_bar_d_mix fix in ci_piecewise above.
library(knitr)

ci_bar_latex <- ci_piecewise %>%
  transmute(
    Bin           = distance_bin,
    `$n_k$`       = n_pairs,
    `$\\widehat{CI}_k$ (bits)` = round(CI_k_v2, 4),
    `$P(d \\in k)$`            = round(P_d, 4),
    Contribution               = round(P_d * CI_k_v2, 4)
  ) %>%
  kable(format = "latex", booktabs = TRUE, escape = FALSE,
        caption = "Piecewise cognitive information, S-based full model.")

writeLines(ci_bar_latex, file.path(out_dir, "ci_bar_table_S.tex"))

# hier_distance analogue, same pattern -- populates a
# Table~\ref{tab:ci-bar-hier}-style placeholder for the structural extension.
# Uses CI_k_hier_v2 (corrected) -- see the H_bar_d_mix_hier fix above.
ci_bar_latex_hier <- ci_piecewise_hier %>%
  transmute(
    Bin           = distance_bin,
    `$n_k$`       = n_pairs,
    `$\\widehat{CI}_k$ (bits)` = round(CI_k_hier_v2, 4),
    `$P(d \\in k)$`            = round(P_d, 4),
    Contribution               = round(P_d * CI_k_hier_v2, 4)
  ) %>%
  kable(format = "latex", booktabs = TRUE, escape = FALSE,
        caption = "Piecewise cognitive information, hier\\_distance-based full model.")

writeLines(ci_bar_latex_hier, file.path(out_dir, "ci_bar_table_hier.tex"))

# --- Key scalars, in case you just want the headline numbers on hand ---
summary_scalars <- tibble(
  quantity = c("CI_bar_S", "eta_bar_S", "CI_bar_hier", "eta_bar_hier",
               "same_vs_heading_agreement"),
  value    = c(CI_bar, eta_bar, CI_bar_hier, eta_bar_hier, same_vs_heading)
)
write_csv(summary_scalars, file.path(out_dir, "summary_scalars.csv"))

message(sprintf("All pipeline outputs written to: %s", normalizePath(out_dir)))
sink()