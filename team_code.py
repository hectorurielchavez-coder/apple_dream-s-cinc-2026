#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but
# you can edit most parts of the required functions, change or remove
# non-required functions, and add your own functions.

################################################################################
#
# Imports — do not remove the ones used by the required functions.
#
################################################################################

import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from itertools import groupby

import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from helper_code import *

################################################################################
# Constants
################################################################################

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Sleep stage constants (CAISR encoding)
EPOCH_SEC        = 30
SAMP_PER_EP      = 60          # 2 Hz × 30 s
VALID_STAGES     = {1, 2, 3, 4, 5}
STAGE_NAMES      = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake"}
MAX_GAP_EPOCHS   = 2           # gaps > 2 epochs → Wake

# Respiratory event codes in resp_caisr
RESP_OA    = 1
RESP_CA    = 2
RESP_MA    = 3
RESP_HY    = 4
RESP_RERA  = 5
RESP_ALL   = {RESP_OA, RESP_CA, RESP_MA, RESP_HY, RESP_RERA}
RESP_APNEA = {RESP_OA, RESP_CA, RESP_MA}
RESP_MIN_DURATION_S = 10       # AASM minimum event duration

################################################################################
#
# Required functions
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Train the ElasticNet model on the full training set."""

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records           = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Extracting features from {num_records} records...')

    features_list = []
    labels_list   = []
    groups_list   = []   # SiteID — used for stratified CV during training

    pbar = tqdm(range(num_records), desc="Extracting", unit="record",
                disable=not verbose)

    for i in pbar:
        try:
            record     = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({"patient": patient_id})

            # ── Label (necesita patient_data de load_demographics) ─────────────
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            label = load_label(patient_data)
            if label not in (0, 1):
                continue

            # ── CAISR annotations ─────────────────────────────────────────────
            algo_file = os.path.join(
                data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
            )
            if not os.path.exists(algo_file):
                if verbose:
                    tqdm.write(f"  ! Missing CAISR file for {patient_id}, skipping")
                continue

            algo_data, _ = load_signal_data(algo_file)
            sleep_feats  = extract_sleep_features(algo_data)   # devuelve dict
            del algo_data

            features_list.append(sleep_feats)
            labels_list.append(label)
            groups_list.append(site_id)

        except Exception as e:
            tqdm.write(f"  !!! Error on record {patient_id}: {e}")
            continue

    pbar.close()

    if len(features_list) == 0:
        raise ValueError("No valid records were processed.")

    # DataFrame con columnas nombradas + __site__ para SiteNormalizer
    X              = pd.DataFrame(features_list)
    X["__site__"]  = groups_list
    y              = np.asarray(labels_list, dtype=int)

    if verbose:
        print(f'Training on {len(y)} records  CI=True:{y.sum()}  CI=False:{(1-y).sum()}')

    # ── ElasticNet (same architecture as refine.py) ───────────────────────────
    model_pipeline = _make_elasticnet(C=0.01, l1_ratio=0.0)
    model_pipeline.fit(X, y)

    # Save
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model_pipeline)

    if verbose:
        print('Model saved.')


def load_model(model_folder, verbose):
    """Load the trained model from disk."""
    model_filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(model_filename)


def run_model(model, record, data_folder, verbose):
    """Run the trained model on a single record and return (binary, probability)."""

    clf = model['model']

    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # ── CAISR annotations ─────────────────────────────────────────────────────
    algo_file = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
    )
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        sleep_feats  = extract_sleep_features(algo_data)   # devuelve dict
    else:
        _KEYS = ["arousal_index", "PLM_index", "pct_REM_first_half",
                 "hypno_entropy", "frag_N3toW", "WASO_min",
                 "cycle_rem_mean", "bout_n_N3"]
        sleep_feats = {k: 0.0 for k in _KEYS}

    # DataFrame con __site__ para que SiteNormalizer aplique correctamente
    features_df             = pd.DataFrame([sleep_feats])
    features_df["__site__"] = site_id

    binary_output      = int(clf.predict(features_df)[0])
    probability_output = float(clf.predict_proba(features_df)[0][1])

    return binary_output, probability_output


################################################################################
#
# Model persistence
#
################################################################################

def save_model(model_folder, model):
    d        = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


################################################################################
#
# ElasticNet pipeline (mirrors refine.py exactly)
#
################################################################################

class _SiteNormalizer(BaseEstimator, TransformerMixin):
    """
    Per-site Z-score normalisation.  Falls back to global stats for unseen sites.
    The __site__ column is consumed here and NOT passed to the classifier.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        feat_cols = [c for c in X.columns if c != "__site__"]
        self.global_median_ = X[feat_cols].median()
        self.global_mean_   = X[feat_cols].mean()
        self.global_std_    = X[feat_cols].std().replace(0, 1)
        self.site_stats_    = {}
        if "__site__" in X.columns:
            for site, grp in X.groupby("__site__"):
                filled = grp[feat_cols].fillna(self.global_median_)
                self.site_stats_[site] = {
                    "mean": filled.mean(),
                    "std":  filled.std().replace(0, 1),
                }
        return self

    def transform(self, X, y=None):
        X         = pd.DataFrame(X).copy()
        feat_cols = [c for c in X.columns if c != "__site__"]
        for col in feat_cols:
            X[col] = X[col].fillna(self.global_median_[col]
                                   if col in self.global_median_ else 0.0)
        result = X[feat_cols].astype(float).copy()
        if self.site_stats_ and "__site__" in X.columns:
            for site, stats in self.site_stats_.items():
                mask = X["__site__"] == site
                if mask.any():
                    result.loc[mask] = (
                        (X.loc[mask, feat_cols] - stats["mean"]) / stats["std"]
                    ).values
            unknown = ~X["__site__"].isin(self.site_stats_)
            if unknown.any():
                for site in X.loc[unknown, "__site__"].unique():
                    m    = X["__site__"] == site
                    sd   = X.loc[m, feat_cols]
                    sm   = sd.mean()
                    ss   = sd.std().replace(0, 1)
                    result.loc[m] = ((sd - sm) / ss).values
        else:
            result = (X[feat_cols] - self.global_mean_) / self.global_std_
        return result.values.astype(float)


def _make_elasticnet(C=0.1, l1_ratio=0.5):
    return Pipeline([
        ("norm", _SiteNormalizer()),
        ("clf",  LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=C, l1_ratio=l1_ratio,
            max_iter=3000, class_weight="balanced",
            random_state=42,
        )),
    ])


################################################################################
#
# Sleep feature extraction  (ported from build_features.py v3 +
#                             temporal_features.py v4 — validated pipeline)
#
################################################################################

# Sentinel: number of features returned by extract_sleep_features()
# (used as fallback dimension in run_model when the CAISR file is absent)
_SLEEP_FEAT_DIM = 46   # updated automatically when the function grows


# ── Stage cleaning (temporal_features.py FIX 4) ──────────────────────────────

def _clean_stages(stages_raw):
    """
    Forward-fill limited to MAX_GAP_EPOCHS; longer gaps → Wake (5).
    Backward-fill at recording start.
    """
    s = np.array(stages_raw, dtype=float)
    n = len(s)
    i = 0
    while i < n:
        v           = s[i]
        is_invalid  = np.isnan(v) or int(v) not in VALID_STAGES
        if not is_invalid:
            i += 1
            continue
        gap_start = i
        while i < n:
            v2 = s[i]
            if np.isnan(v2) or int(v2) not in VALID_STAGES:
                i += 1
            else:
                break
        gap_len    = i - gap_start
        prev_stage = None
        for k in range(gap_start - 1, -1, -1):
            vk = s[k]
            if not np.isnan(vk) and int(vk) in VALID_STAGES:
                prev_stage = int(vk)
                break
        fill_val = 5 if (prev_stage is None or gap_len > MAX_GAP_EPOCHS) else prev_stage
        s[gap_start:gap_start + gap_len] = fill_val

    first_valid = next(
        (int(v) for v in s if not np.isnan(v) and int(v) in VALID_STAGES), 5
    )
    for i in range(n):
        if np.isnan(s[i]) or int(s[i]) not in VALID_STAGES:
            s[i] = first_valid
        else:
            break
    return s.astype(int)


def _extract_stages_from_caisr(algo_data):
    """
    Read stage_caisr from the algo_data dict (already loaded by helper_code).
    Uses median of central chunk per epoch (FIX 5).
    Returns cleaned integer stage array.
    """
    stage_key = next(
        (k for k in algo_data if "stage_caisr" in k.lower()), None
    )
    if stage_key is None:
        return np.array([], dtype=int)

    sig      = algo_data[stage_key].astype(float)
    n_epochs = len(sig) // SAMP_PER_EP

    # FIX 5: median of central third of each epoch
    c_start = SAMP_PER_EP // 3
    c_end   = 2 * SAMP_PER_EP // 3

    stages_raw = np.array([
        int(np.round(np.median(
            sig[ep * SAMP_PER_EP + c_start: ep * SAMP_PER_EP + c_end]
        )))
        for ep in range(n_epochs)
    ], dtype=float)

    stages_raw[~np.isin(stages_raw, [1, 2, 3, 4, 5, 9])] = np.nan
    return _clean_stages(stages_raw)


# ── Respiratory event counting (build_features.py FIX 6+7) ──────────────────

def _count_respiratory_events(resp, query_codes,
                               min_duration_s=RESP_MIN_DURATION_S):
    """
    Anti-flicker multiclass counting with AASM ≥10 s duration filter.
    Opens on any RESP_ALL code, closes on 0, classifies by dominant code.
    """
    count        = 0
    in_event     = False
    event_start  = 0
    event_buffer = []

    for i, v in enumerate(resp):
        if v in RESP_ALL:
            if not in_event:
                in_event     = True
                event_start  = i
                event_buffer = [v]
            else:
                event_buffer.append(v)
        else:
            if in_event:
                duration = i - event_start
                if duration >= min_duration_s:
                    dominant = Counter(event_buffer).most_common(1)[0][0]
                    if dominant in query_codes:
                        count += 1
                in_event     = False
                event_buffer = []

    # Event reaching end of array
    if in_event and event_buffer:
        duration = len(resp) - event_start
        if duration >= min_duration_s:
            dominant = Counter(event_buffer).most_common(1)[0][0]
            if dominant in query_codes:
                count += 1
    return count


def _count_events_simple(arr, target_vals):
    """Rising-edge counter for binary signals (arousal, limb)."""
    count    = 0
    in_event = False
    for v in arr:
        if v in target_vals and not in_event:
            count   += 1
            in_event = True
        elif v not in target_vals:
            in_event = False
    return count


# ── Sleep architecture features (build_features.py v3) ───────────────────────

def _compute_architecture_features(stages, algo_data):
    """
    Returns a dict of clinical sleep metrics.
    AHI denominator = TST (PSG standard — FIX 8).
    """
    feat    = {}
    n_total = len(stages)

    # Time in bed / total sleep time
    tib_epochs   = n_total
    sleep_epochs = np.isin(stages, [1, 2, 3, 4]).sum()
    tib_min      = tib_epochs   * EPOCH_SEC / 60
    tst_min      = sleep_epochs * EPOCH_SEC / 60
    tst_h        = tst_min / 60
    tib_h        = tib_min / 60

    feat["TIB_min"] = tib_min
    feat["TST_min"] = tst_min
    feat["SE_pct"]  = (tst_min / tib_min * 100) if tib_min > 0 else 0.0

    # Stage percentages
    n_valid = np.isin(stages, list(STAGE_NAMES.keys())).sum()
    for code, name in STAGE_NAMES.items():
        n = np.sum(stages == code)
        feat[f"pct_{name}"] = (n / n_valid * 100) if n_valid > 0 else 0.0

    # SOL
    sol = np.nan
    for i, s in enumerate(stages):
        if s in {1, 2, 3, 4}:
            sol = i * EPOCH_SEC / 60
            break
    feat["SOL_min"] = sol if not np.isnan(sol) else tib_min

    # WASO
    first_sleep = next((i for i, s in enumerate(stages) if s in {1,2,3,4}), None)
    last_sleep  = next((i for i, s in enumerate(reversed(stages)) if s in {1,2,3,4}), None)
    waso        = 0.0
    if first_sleep is not None and last_sleep is not None:
        last_idx = n_total - 1 - last_sleep
        middle   = stages[first_sleep:last_idx + 1]
        waso     = np.sum(middle == 5) * EPOCH_SEC / 60
    feat["WASO_min"] = waso

    # REM latency (AASM: count only sleep epochs before first REM)
    rem_lat = np.nan
    if first_sleep is not None:
        sleep_before_rem = 0
        for i in range(first_sleep, n_total):
            if stages[i] == 4:
                rem_lat = sleep_before_rem * EPOCH_SEC / 60
                break
            elif stages[i] in {1, 2, 3}:
                sleep_before_rem += 1
    feat["REM_latency_min"] = rem_lat if not np.isnan(rem_lat) else tst_min

    # Arousal index (denominator = TIB, no EEG-gating)
    ar_key   = next((k for k in algo_data if "arousal_caisr" in k.lower()), None)
    n_arous  = 0
    if ar_key is not None:
        ar_sig  = np.clip(np.round(algo_data[ar_key]).astype(int), 0, 1)
        n_arous = int((np.diff(ar_sig.astype(int)) > 0).sum())
    feat["arousal_index"] = n_arous / tib_h if tib_h > 0 else 0.0

    # AHI — FIX 6+7+8
    resp_key = next((k for k in algo_data if "resp_caisr" in k.lower()), None)
    if resp_key is not None:
        resp          = algo_data[resp_key].astype(int)
        n_ahi_events  = _count_respiratory_events(resp, {RESP_OA,RESP_CA,RESP_MA,RESP_HY})
        n_oa          = _count_respiratory_events(resp, {RESP_OA})
        n_ca          = _count_respiratory_events(resp, {RESP_CA})
        n_hy          = _count_respiratory_events(resp, {RESP_HY})
        n_rera        = _count_respiratory_events(resp, {RESP_RERA})
    else:
        n_ahi_events = n_oa = n_ca = n_hy = n_rera = 0

    feat["AHI"]        = n_ahi_events / tst_h if tst_h > 0 else 0.0
    feat["AI_obst"]    = n_oa          / tst_h if tst_h > 0 else 0.0
    feat["AI_cent"]    = n_ca          / tst_h if tst_h > 0 else 0.0
    feat["AI_hyp"]     = n_hy          / tst_h if tst_h > 0 else 0.0
    feat["RERA_index"] = n_rera        / tst_h if tst_h > 0 else 0.0

    # PLM index (denominator = TIB)
    limb_key = next((k for k in algo_data if "limb_caisr" in k.lower()), None)
    if limb_key is not None:
        limb      = algo_data[limb_key].astype(int)
        n_plm     = _count_events_simple(limb, {2})
    else:
        n_plm = 0
    feat["PLM_index"] = n_plm / tib_h if tib_h > 0 else 0.0

    return feat


# ── Temporal / complexity features (temporal_features.py v4) ─────────────────

def _get_bouts(stages):
    return [(s, sum(1 for _ in g)) for s, g in groupby(stages)]


def _shannon_entropy(stages):
    counts = Counter(int(s) for s in stages if s in STAGE_NAMES)
    total  = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _transition_entropy_cross(stages):
    valid = [int(s) for s in stages if s in STAGE_NAMES]
    cross = Counter((a, b) for a, b in zip(valid[:-1], valid[1:]) if a != b)
    total = sum(cross.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in cross.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _lz_complexity(stages):
    seq = "".join(str(int(s)) for s in stages)
    n   = len(seq)
    if n < 2:
        return 0.0
    i, c, q, r = 0, 1, seq[0], seq[0]
    while i < n - 1:
        i += 1
        if seq[i] not in (q + r[:-1]):
            c += 1
            q, r = r, seq[i]
        else:
            r += seq[i]
    alph = len(set(stages.tolist()))
    if alph <= 1:
        return 0.0
    norm = (n / np.log2(n)) * np.log2(alph)
    return float(c / norm) if norm > 0 else 0.0


def _compute_temporal_features(stages):
    """Bout stats, entropy, fragmentation metrics."""
    feat  = {}
    bouts = _get_bouts(stages)

    # Bout statistics per stage
    for code, name in STAGE_NAMES.items():
        durs = [n * EPOCH_SEC / 60 for s, n in bouts if s == code]
        feat[f"bout_mean_{name}"]   = float(np.mean(durs))   if durs else 0.0
        feat[f"bout_n_{name}"]      = len(durs)

    # Complexity
    valid = stages[np.isin(stages, list(STAGE_NAMES.keys()))]
    feat["hypno_entropy"]            = _shannon_entropy(valid)
    feat["hypno_lz_complexity"]      = _lz_complexity(valid)
    feat["transition_entropy_cross"] = _transition_entropy_cross(valid)

    # Stage entropy (FIX 6 replacement for rem_progression)
    counts = np.array([np.sum(valid == c) for c in STAGE_NAMES], dtype=float)
    total  = counts.sum()
    if total > 0:
        probs = counts / total
        feat["stage_entropy"] = float(
            -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        )
    else:
        feat["stage_entropy"] = 0.0

    # REM fragmentation (FIX 6)
    rem_bouts      = [n for s, n in bouts if s == 4]
    tst_h          = np.sum(np.isin(stages, [1,2,3,4])) * EPOCH_SEC / 3600
    feat["rem_fragmentation"]  = len(rem_bouts) / tst_h if tst_h > 0 else 0.0
    feat["rem_bout_mean_min"]  = float(np.mean([n * EPOCH_SEC / 60
                                                for n in rem_bouts])) \
                                 if rem_bouts else 0.0

    # N3 fragmentation (CAP proxy)
    n3_bouts = [n for s, n in bouts if s == 1]
    n3_short = sum(1 for n in n3_bouts if n <= 2)
    n3_long  = sum(1 for n in n3_bouts if n > 10)
    feat["n3_bouts_short"]         = n3_short
    feat["n3_bouts_long"]          = n3_long
    feat["n3_fragmentation_ratio"] = (n3_short / len(n3_bouts)
                                      if n3_bouts else 0.0)

    # Temporal distribution (N3 front-loading, REM back-loading)
    n        = len(valid)
    half     = n // 2
    first_h  = valid[:half]
    second_h = valid[half:]

    def pct(arr, code):
        return float(np.sum(arr == code) / len(arr) * 100) if len(arr) else 0.0

    n3_first   = pct(first_h,  1)
    n3_total   = pct(valid,    1)
    rem_second = pct(second_h, 4)
    rem_total  = pct(valid,    4)
    feat["n3_front_loading"]  = (n3_first   / n3_total)  if n3_total  > 5.0 else 0.0
    feat["rem_back_loading"]  = (rem_second / rem_total) if rem_total > 0   else 0.0

    return feat


# ── Master feature extractor ──────────────────────────────────────────────────

# Ordered feature names — must match the order features are assembled below.
# Update _SLEEP_FEAT_DIM if you add / remove features here.
_SLEEP_FEATURE_NAMES = [
    # Architecture (14)
    "TIB_min", "TST_min", "SE_pct",
    "pct_N3", "pct_N2", "pct_N1", "pct_REM", "pct_Wake",
    "SOL_min", "WASO_min", "REM_latency_min",
    "arousal_index", "AHI", "PLM_index",
    # Additional event indices (4)
    "AI_obst", "AI_cent", "AI_hyp", "RERA_index",
    # Bout stats (10: 5 stages × mean + n)
    "bout_mean_N3", "bout_n_N3",
    "bout_mean_N2", "bout_n_N2",
    "bout_mean_N1", "bout_n_N1",
    "bout_mean_REM", "bout_n_REM",
    "bout_mean_Wake", "bout_n_Wake",
    # Complexity (4)
    "hypno_entropy", "hypno_lz_complexity",
    "transition_entropy_cross", "stage_entropy",
    # REM fragmentation (2)
    "rem_fragmentation", "rem_bout_mean_min",
    # N3 fragmentation (3)
    "n3_bouts_short", "n3_bouts_long", "n3_fragmentation_ratio",
    # Temporal distribution (2)
    "n3_front_loading", "rem_back_loading",
    # Prob channels (3)
    "prob_w_mean", "prob_n3_mean", "prob_arous_mean",
]

_SLEEP_FEAT_DIM = len(_SLEEP_FEATURE_NAMES)   # = 42 — kept in sync



# ── Top-8 feature extractor (refine.py validated subset) ────────────────────

_TOP8_KEYS = [
    "arousal_index", "PLM_index", "pct_REM_first_half",
    "hypno_entropy", "frag_N3toW", "WASO_min",
    "cycle_rem_mean", "bout_n_N3",
]

def _get_top_8_features(stages, algo_data):
    """Extrae exactamente las 8 features del modelo refinado."""
    feat = {}

    # WASO, arousal_index, PLM_index
    arch = _compute_architecture_features(stages, algo_data)
    feat["WASO_min"]      = arch["WASO_min"]
    feat["arousal_index"] = arch["arousal_index"]
    feat["PLM_index"]     = arch["PLM_index"]

    # hypno_entropy, bout_n_N3
    temp = _compute_temporal_features(stages)
    feat["hypno_entropy"] = temp["hypno_entropy"]
    feat["bout_n_N3"]     = temp["bout_n_N3"]

    # pct_REM_first_half
    valid   = stages[np.isin(stages, list(STAGE_NAMES.keys()))]
    first_h = valid[:len(valid) // 2]
    feat["pct_REM_first_half"] = (
        float(np.sum(first_h == 4) / len(first_h) * 100) if len(first_h) else 0.0
    )

    # frag_N3toW (transición directa N3→W)
    trans = Counter(zip(valid[:-1], valid[1:]))
    feat["frag_N3toW"] = float(trans.get((1, 5), 0))

    # cycle_rem_mean (Feinberg & Floyd, min_nrem=6, min_rem=3)
    s = np.where(np.isin(stages, [1, 2, 3]), 1,
        np.where(stages == 4, 2, 3))
    cycles, i, n = [], 0, len(s)
    while i < n:
        if s[i] != 1:
            i += 1
            continue
        nrem_count, j = 0, i
        while j < n:
            if s[j] == 1:
                nrem_count += 1
                j += 1
            elif s[j] == 3:
                k = j
                while k < n and s[k] == 3:
                    k += 1
                if (k - j) < 5 and k < n and s[k] == 1:
                    j = k
                else:
                    break
            else:
                break
        if nrem_count < 6:
            i = j + 1
            continue
        while j < n and s[j] == 3:
            j += 1
        if j >= n or s[j] != 2:
            i = j + 1
            continue
        rem_count = 0
        while j < n and s[j] == 2:
            rem_count += 1
            j += 1
        if rem_count < 3:
            i = j + 1
            continue
        cycles.append(rem_count)
        i = j
    feat["cycle_rem_mean"] = (
        float(np.mean([c * EPOCH_SEC / 60 for c in cycles])) if cycles else 0.0
    )

    return feat

def extract_sleep_features(algo_data):
    """
    Devuelve un dict con exactamente las 8 features del modelo refinado.
    El SiteNormalizer espera un DataFrame con columnas nombradas — este dict
    se convierte a DataFrame en train_model y run_model antes de fit/predict.
    """
    stages = _extract_stages_from_caisr(algo_data)

    if len(stages) < 10:
        return {k: 0.0 for k in _TOP8_KEYS}

    raw = _get_top_8_features(stages, algo_data)
    # Garantizar el orden de columnas y limpiar NaN/inf
    result = {}
    for k in _TOP8_KEYS:
        v = raw.get(k, 0.0)
        result[k] = float(v) if np.isfinite(v) else 0.0
    return result


################################################################################
#
# Physiological signal features  (unchanged from original team_code.py template)
#
################################################################################

def extract_demographic_features(data):
    age      = np.array([load_age(data)])
    sex      = load_sex(data)
    sex_vec  = np.zeros(3)
    if sex == 'Female': sex_vec[0] = 1
    elif sex == 'Male': sex_vec[1] = 1
    else:               sex_vec[2] = 1
    race_category = get_standardized_race(data).lower()
    race_vec      = np.zeros(5)
    race_mapping  = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1
    bmi = np.array([load_bmi(data)])
    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs,
                                    csv_path=DEFAULT_CSV_PATH):
    original_labels = list(physiological_data.keys())
    rename_rules    = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    processed_channels = {}
    processed_fs       = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]

    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-']),
    ]
    for target, pos, neg_list in bipolar_configs:
        if target in processed_channels or pos not in processed_channels:
            continue
        if not all(n in processed_channels for n in neg_list):
            continue
        all_involved = [pos] + neg_list
        fs_values    = [processed_fs[ch] for ch in all_involved if ch in processed_fs]
        if len(set(fs_values)) > 1:
            continue
        ref = (processed_channels[neg_list[0]] if len(neg_list) == 1
               else tuple(processed_channels[n] for n in neg_list))
        derived = derive_bipolar_signal(processed_channels[pos], ref)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target]       = processed_fs.get(pos, 0)

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'],
    }

    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                break

        if sig is not None and len(sig) > 1:
            std_val    = np.std(sig)
            mav_val    = np.mean(np.abs(sig))
            zcr        = np.mean(np.diff(np.sign(sig)) != 0)
            rms        = np.sqrt(np.mean(sig**2))
            activity   = np.var(sig)
            diff_sig   = np.diff(sig)
            mobility   = (np.sqrt(np.var(diff_sig) / activity)
                          if activity > 0 else 0.0)
            diff2_sig  = np.diff(diff_sig)
            var_d2     = np.var(diff2_sig)
            var_d1     = np.var(diff_sig)
            complexity = ((np.sqrt(var_d2 / var_d1) / mobility)
                          if (var_d1 > 0 and mobility > 0) else 0.0)
            final_features.extend([std_val, mav_val, zcr, rms,
                                    activity, mobility, complexity])
        else:
            final_features.extend([0.0] * 7)

    return np.array(final_features, dtype=np.float32)