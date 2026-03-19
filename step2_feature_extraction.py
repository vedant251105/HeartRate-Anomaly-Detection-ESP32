import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
os.chdir(r'D:\CTAIML FINAL WORK')
print(f"Working directory: {os.getcwd()}")
pamap2_df = pd.read_pickle('pamap2_cleaned.pkl')
ppg_df    = pd.read_pickle('ppg_dalia_cleaned.pkl')

print(f"PAMAP2 loaded   : {len(pamap2_df):,} rows")
print(f"PPG-DaLiA loaded: {len(ppg_df):,} windows")

PAMAP2_FS  = 100
WINDOW_SEC = 5
WINDOW_LEN = PAMAP2_FS * WINDOW_SEC  # 500 samples
STEP       = 50                       # 0.5 sec step → maximum windows

activity_code_map = {'rest':0, 'walk':1, 'run':2, 'intense':3}
EXPECTED_HR       = {0:70, 1:95, 2:140, 3:155}

# Run this diagnostic first — add to top of step2 temporarily
print("PAMAP2 HR NaN analysis:")
for sid in pamap2_df.subject_id.unique():
    sub = pamap2_df[pamap2_df.subject_id == sid]
    total = len(sub)
    nan_count = sub['heart_rate'].isna().sum()
    print(f"  subject{sid}: {total:,} rows, "
          f"{nan_count:,} NaN HR ({100*nan_count/total:.0f}%)")

def extract_features_pamap2(df):
    all_features, all_labels, all_hr_raw = [], [], []

    for sid in df.subject_id.unique():
        sub = df[df.subject_id == sid].reset_index(drop=True)

        # Personal baseline
        personal_stats = {}
        for act in sub.activity.unique():
            act_hr = sub[sub.activity == act]['heart_rate'].values
            personal_stats[act] = {
                'mean': np.mean(act_hr),
                'std' : max(np.std(act_hr), 1.0)
            }

        # Process each activity segment separately
        # This avoids windows that straddle two activities
        for act_label in sub.activity.unique():
            act_df = sub[sub.activity == act_label].reset_index(drop=True)

            if len(act_df) < WINDOW_LEN:
                continue  # segment too short

            act_c  = activity_code_map.get(act_label, 0)
            p_mean = personal_stats[act_label]['mean']
            p_std  = personal_stats[act_label]['std']
            prev_hr_mean = None

            for start in range(0, len(act_df) - WINDOW_LEN, STEP):
                w = act_df.iloc[start : start + WINDOW_LEN]

                hr_arr = w['heart_rate'].values

                # Skip if too many NaN in HR
                valid = hr_arr[~np.isnan(hr_arr)]
                if len(valid) < WINDOW_LEN * 0.8:
                    continue

                hr_arr   = np.where(np.isnan(hr_arr), np.nanmean(hr_arr), hr_arr)
                hr_mean  = np.mean(hr_arr)
                hr_std   = np.std(hr_arr)
                hr_zscore = (hr_mean - p_mean) / p_std

                rr_ms = (60.0 / (hr_arr + 1e-6)) * 1000.0
                rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 1 else 0.0

                ax = w['hand_acc_x'].values
                ay = w['hand_acc_y'].values
                az = w['hand_acc_z'].values
                mag    = np.sqrt(ax**2 + ay**2 + az**2)
                a_mean = np.mean(mag)
                a_std  = np.std(mag)

                hr_delta = (hr_mean - prev_hr_mean) if prev_hr_mean is not None else 0.0
                prev_hr_mean = hr_mean

                expected = EXPECTED_HR[act_c]
                activity_hr_ratio = (hr_mean - expected) / (p_std + 1.0)

                features = np.array([
                    hr_zscore, hr_std, rmssd, float(act_c),
                    a_mean, a_std, hr_delta, activity_hr_ratio
                ], dtype=np.float32)

                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue

                all_features.append(features)
                all_labels.append(sid)
                all_hr_raw.append(hr_mean)

        print(f"  subject{sid} → {sum(1 for l in all_labels if l==sid):,} windows")

    return np.array(all_features), np.array(all_labels), np.array(all_hr_raw)


def extract_features_ppg(df):
    all_features, all_labels, all_hr_raw = [], [], []

    PPG_ACTIVITY_MAP = {
        1:'rest', 2:'walk', 3:'intense',
        4:'intense', 5:'rest', 6:'rest', 7:'walk'
    }

    for sid in df.subject_id.unique():
        sub = df[df.subject_id == sid].reset_index(drop=True)

        personal_stats = {}
        for act in sub.activity.unique():
            act_hr = sub[sub.activity == act]['heart_rate'].values
            personal_stats[act] = {
                'mean': np.mean(act_hr),
                'std' : max(np.std(act_hr), 1.0)
            }

        prev_hr_mean = None

        for _, row in sub.iterrows():
            act_label = row['activity']
            act_c     = activity_code_map.get(act_label, 0)
            hr_mean   = row['heart_rate']

            p_mean    = personal_stats[act_label]['mean']
            p_std     = personal_stats[act_label]['std']
            hr_zscore = (hr_mean - p_mean) / p_std
            hr_std    = p_std * 0.3
            rr_ms     = (60.0 / (hr_mean + 1e-6)) * 1000.0
            rmssd     = abs(rr_ms - (60000.0 / (p_mean + 1e-6)))

            a_mean    = row['a_mean']
            a_std     = row['a_std']
            hr_delta  = (hr_mean - prev_hr_mean) if prev_hr_mean is not None else 0.0
            prev_hr_mean = hr_mean

            expected  = EXPECTED_HR[act_c]
            activity_hr_ratio = (hr_mean - expected) / (p_std + 1.0)

            features = np.array([
                hr_zscore, hr_std, rmssd, float(act_c),
                a_mean, a_std, hr_delta, activity_hr_ratio
            ], dtype=np.float32)

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue

            all_features.append(features)
            all_labels.append(1000 + sid)
            all_hr_raw.append(hr_mean)

    return np.array(all_features), np.array(all_labels), np.array(all_hr_raw)


# ── Run ───────────────────────────────────────────────────────
print("\nExtracting features from PAMAP2...")
X_pam, sub_pam, hr_pam = extract_features_pamap2(pamap2_df)
print(f"  Total PAMAP2 windows : {len(X_pam):,}")

print("\nExtracting features from PPG-DaLiA...")
X_ppg, sub_ppg, hr_ppg = extract_features_ppg(ppg_df)
print(f"  Total PPG-DaLiA windows : {len(X_ppg):,}")

# ── Combine ───────────────────────────────────────────────────
X_all   = np.vstack([X_pam, X_ppg])
sub_all = np.concatenate([sub_pam, sub_ppg])
hr_all  = np.concatenate([hr_pam, hr_ppg])

print(f"\nCombined dataset:")
print(f"  Total windows  : {len(X_all):,}")
print(f"  Feature shape  : {X_all.shape}")
print(f"  Unique subjects: {len(np.unique(sub_all))}")
print(f"  HR range       : {hr_all.min():.0f}–{hr_all.max():.0f} BPM")

# ── Normalize ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"\nScaler means : {scaler.mean_.round(3)}")
print(f"Scaler stds  : {scaler.scale_.round(3)}")

# ── Plot ──────────────────────────────────────────────────────
feature_names = ['hr_zscore','hr_std','rmssd','activity',
                 'a_mean','a_std','hr_delta','activity_hr_ratio']

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, (ax, name) in enumerate(zip(axes.flatten(), feature_names)):
    ax.hist(X_scaled[:, i], bins=50, color='steelblue', alpha=0.7)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('scaled value')
plt.suptitle('Feature distributions — should be roughly bell-shaped')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100)
plt.show()

# ── Diagnostic — run once then delete ────────────────────────
print("\nPAMAP2 activity segment length analysis:")
for sid in pamap2_df.subject_id.unique():
    sub = pamap2_df[pamap2_df.subject_id == sid].reset_index(drop=True)
    sub['act_block'] = (sub['activity'] != sub['activity'].shift()).cumsum()
    segments = sub.groupby('act_block')['activity'].agg(['first','count'])
    short = segments[segments['count'] < 500]
    long  = segments[segments['count'] >= 500]
    print(f"  subject{sid}: {len(segments)} segments | "
          f"{len(long)} usable (>=500) | "
          f"{len(short)} too short (<500) | "
          f"avg length: {segments['count'].mean():.0f} rows")

# ── Save ──────────────────────────────────────────────────────
np.save('X_scaled.npy',     X_scaled)
np.save('X_raw.npy',        X_all)
np.save('subjects.npy',     sub_all)
np.save('hr_raw.npy',       hr_all)
np.save('scaler_mean.npy',  scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

print("\nStep 2 complete ✓")
print(f"  X_scaled.npy saved — {len(X_all):,} windows × 8 features")