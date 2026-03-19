import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import pickle

os.chdir(r'D:\CTAIML FINAL WORK')

# ── Load cleaned datasets ─────────────────────────────────────
pamap2_df = pd.read_pickle('pamap2_cleaned.pkl')
ppg_df    = pd.read_pickle('ppg_dalia_cleaned.pkl')
X_raw     = np.load('X_raw.npy')
X_scaled  = np.load('X_scaled.npy')
hr_raw    = np.load('hr_raw.npy')
subjects  = np.load('subjects.npy')

feature_names = ['HR Z-Score', 'HR Std', 'RMSSD',
                 'Activity Code', 'Accel Mean',
                 'Accel Std', 'HR Delta', 'Activity HR Ratio']

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── 1. Dataset overview ───────────────────────────────────────
print("\n--- Dataset Overview ---")
print(f"PAMAP2   : {len(pamap2_df):,} rows | "
      f"{pamap2_df.subject_id.nunique()} subjects")
print(f"PPG-DaLiA: {len(ppg_df):,} windows | "
      f"{ppg_df.subject_id.nunique()} subjects")
print(f"Combined : {len(X_raw):,} windows | "
      f"{len(np.unique(subjects))} subjects")
print(f"Features : {X_raw.shape[1]}")

print("\nPAMAP2 HR stats:")
print(pamap2_df['heart_rate'].describe().round(2))

print("\nPPG-DaLiA HR stats:")
print(ppg_df['heart_rate'].describe().round(2))

# ── 2. HR distribution per subject ───────────────────────────
print("\n--- Per Subject HR Stats ---")
print(f"{'Subject':>10} {'Min':>8} {'Max':>8} "
      f"{'Mean':>8} {'Std':>8} {'Dataset':>10}")

for sid in sorted(pamap2_df.subject_id.unique()):
    hr = pamap2_df[pamap2_df.subject_id==sid]['heart_rate']
    print(f"  {sid:>8} {hr.min():>8.1f} {hr.max():>8.1f} "
          f"{hr.mean():>8.1f} {hr.std():>8.1f} {'PAMAP2':>10}")

for sid in sorted(ppg_df.subject_id.unique()):
    hr = ppg_df[ppg_df.subject_id==sid]['heart_rate']
    print(f"  S{sid:>7} {hr.min():>8.1f} {hr.max():>8.1f} "
          f"{hr.mean():>8.1f} {hr.std():>8.1f} {'PPG-DaLiA':>10}")

# ── 3. Activity distribution ──────────────────────────────────
print("\n--- Activity Distribution ---")
print("PAMAP2:")
print(pamap2_df['activity'].value_counts())
print("\nPPG-DaLiA:")
print(ppg_df['activity'].value_counts())

# ── 4. Feature statistics ─────────────────────────────────────
print("\n--- Feature Statistics (raw, before scaling) ---")
df_feat = pd.DataFrame(X_raw, columns=feature_names)
print(df_feat.describe().round(3))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

# ── Plot 1: HR distribution by dataset ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Heart Rate Distribution Across Subjects',
             fontsize=14, fontweight='bold')

for sid in pamap2_df.subject_id.unique():
    hr = pamap2_df[pamap2_df.subject_id==sid]['heart_rate']
    axes[0].hist(hr, bins=30, alpha=0.4, label=f'S{sid}')
axes[0].set_title('PAMAP2 — HR per subject')
axes[0].set_xlabel('Heart Rate (BPM)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=6, ncol=3)
axes[0].axvline(pamap2_df['heart_rate'].mean(), color='red',
                linestyle='--', linewidth=2, label='Mean')

for sid in ppg_df.subject_id.unique():
    hr = ppg_df[ppg_df.subject_id==sid]['heart_rate']
    axes[1].hist(hr, bins=20, alpha=0.4, label=f'S{sid}')
axes[1].set_title('PPG-DaLiA — HR per subject')
axes[1].set_xlabel('Heart Rate (BPM)')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=6, ncol=4)

plt.tight_layout()
plt.savefig('eda_01_hr_distribution.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 2: HR by activity (box plot) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Heart Rate by Activity Level',
             fontsize=14, fontweight='bold')

pamap2_df.boxplot(column='heart_rate', by='activity',
                   ax=axes[0], patch_artist=True)
axes[0].set_title('PAMAP2')
axes[0].set_xlabel('Activity')
axes[0].set_ylabel('HR (BPM)')
plt.sca(axes[0])
plt.xticks(rotation=15)

ppg_df.boxplot(column='heart_rate', by='activity',
                ax=axes[1], patch_artist=True)
axes[1].set_title('PPG-DaLiA')
axes[1].set_xlabel('Activity')
axes[1].set_ylabel('HR (BPM)')

plt.suptitle('')
plt.tight_layout()
plt.savefig('eda_02_hr_by_activity.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 3: Activity sample count ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Activity Distribution in Datasets',
             fontsize=14, fontweight='bold')

p_counts = pamap2_df['activity'].value_counts()
axes[0].bar(p_counts.index, p_counts.values,
            color=['#2196F3','#4CAF50','#FF9800','#F44336'])
axes[0].set_title('PAMAP2')
axes[0].set_xlabel('Activity')
axes[0].set_ylabel('Sample count')
for i, (act, cnt) in enumerate(p_counts.items()):
    axes[0].text(i, cnt + 100, str(cnt), ha='center', fontsize=9)

q_counts = ppg_df['activity'].value_counts()
axes[1].bar(q_counts.index, q_counts.values,
            color=['#2196F3','#4CAF50','#FF9800'])
axes[1].set_title('PPG-DaLiA')
axes[1].set_xlabel('Activity')
axes[1].set_ylabel('Sample count')
for i, (act, cnt) in enumerate(q_counts.items()):
    axes[1].text(i, cnt + 5, str(cnt), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_03_activity_distribution.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 4: Feature distributions after scaling ───────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Feature Distributions After StandardScaler',
             fontsize=14, fontweight='bold')

colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0',
          '#F44336','#00BCD4','#FF5722','#607D8B']

for i, (ax, name, color) in enumerate(
        zip(axes.flatten(), feature_names, colors)):
    ax.hist(X_scaled[:, i], bins=50, color=color,
            alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--',
               linewidth=1, alpha=0.7)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Scaled value', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.text(0.98, 0.95,
            f'μ={X_scaled[:,i].mean():.2f}\n'
            f'σ={X_scaled[:,i].std():.2f}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.7))

plt.tight_layout()
plt.savefig('eda_04_feature_distributions.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 5: Correlation heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Feature Correlation Matrix',
             fontsize=14, fontweight='bold')

df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
corr = df_scaled.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=12)

plt.tight_layout()
plt.savefig('eda_05_correlation_heatmap.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 6: HR variability across subjects ────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('HR Mean ± Std per Subject (Both Datasets)',
             fontsize=14, fontweight='bold')

subject_ids   = []
hr_means      = []
hr_stds       = []
dataset_labels = []

for sid in sorted(pamap2_df.subject_id.unique()):
    hr = pamap2_df[pamap2_df.subject_id==sid]['heart_rate']
    subject_ids.append(f'P{sid}')
    hr_means.append(hr.mean())
    hr_stds.append(hr.std())
    dataset_labels.append('PAMAP2')

for sid in sorted(ppg_df.subject_id.unique()):
    hr = ppg_df[ppg_df.subject_id==sid]['heart_rate']
    subject_ids.append(f'D-S{sid}')
    hr_means.append(hr.mean())
    hr_stds.append(hr.std())
    dataset_labels.append('PPG-DaLiA')

hr_means = np.array(hr_means)
hr_stds  = np.array(hr_stds)
x        = np.arange(len(subject_ids))

colors_bar = ['#2196F3' if d == 'PAMAP2'
              else '#4CAF50' for d in dataset_labels]

bars = ax.bar(x, hr_means, yerr=hr_stds, color=colors_bar,
              alpha=0.8, capsize=4, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(subject_ids, rotation=45,
                   ha='right', fontsize=8)
ax.set_ylabel('Heart Rate (BPM)')
ax.set_xlabel('Subject ID')
ax.axhline(np.mean(hr_means), color='red', linestyle='--',
           linewidth=2, label=f'Overall mean: '
                              f'{np.mean(hr_means):.1f} BPM')
ax.legend()

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', label='PAMAP2'),
    Patch(facecolor='#4CAF50', label='PPG-DaLiA')
]
ax.legend(handles=legend_elements +
          [plt.Line2D([0],[0], color='red', linestyle='--',
                      label=f'Overall mean: '
                            f'{np.mean(hr_means):.1f} BPM')])

plt.tight_layout()
plt.savefig('eda_06_hr_per_subject.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Plot 7: Pairplot of key features ─────────────────────────
print("\nGenerating pairplot (may take 30 seconds)...")
df_pair = pd.DataFrame(X_scaled[:, [0, 1, 2, 4]],
                       columns=['HR Z-Score', 'HR Std',
                                'RMSSD', 'Accel Mean'])

# Add activity label for coloring
activity_labels = ['rest' if X_raw[i,3]==0
                   else 'walk' if X_raw[i,3]==1
                   else 'run'  if X_raw[i,3]==2
                   else 'intense'
                   for i in range(len(X_raw))]
df_pair['Activity'] = activity_labels

pair_plot = sns.pairplot(df_pair, hue='Activity',
                          plot_kws={'alpha': 0.4, 's': 10},
                          palette={'rest':'#2196F3',
                                   'walk':'#4CAF50',
                                   'run':'#FF9800',
                                   'intense':'#F44336'})
pair_plot.fig.suptitle('Pairplot of Key Features by Activity',
                        y=1.02, fontsize=13, fontweight='bold')
plt.savefig('eda_07_pairplot.png', dpi=100,
            bbox_inches='tight')
plt.show()

# ── Plot 8: HR zscore shows person-agnostic normalization ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Person-Agnostic Normalization — '
             'Raw HR vs HR Z-Score',
             fontsize=13, fontweight='bold')

# Raw HR by subject
for sid in np.unique(subjects)[:9]:
    mask = subjects == sid
    axes[0].hist(hr_raw[mask], bins=20, alpha=0.5)
axes[0].set_title('Raw HR (different for each person)')
axes[0].set_xlabel('Heart Rate (BPM)')
axes[0].set_ylabel('Count')
axes[0].text(0.05, 0.95,
             'Each person has different\nHR distribution',
             transform=axes[0].transAxes,
             va='top', fontsize=9,
             bbox=dict(boxstyle='round',
                       facecolor='lightyellow'))

# HR zscore by subject — should overlap
for sid in np.unique(subjects)[:9]:
    mask = subjects == sid
    axes[1].hist(X_scaled[mask, 0], bins=20, alpha=0.5)
axes[1].set_title('HR Z-Score (same distribution for everyone)')
axes[1].set_xlabel('Z-Score (standard deviations from personal mean)')
axes[1].set_ylabel('Count')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].text(0.05, 0.95,
             'All subjects normalized\nto same distribution',
             transform=axes[1].transAxes,
             va='top', fontsize=9,
             bbox=dict(boxstyle='round',
                       facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('eda_08_normalization.png', dpi=120,
            bbox_inches='tight')
plt.show()

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("EDA SUMMARY")
print("=" * 55)
print(f"Total windows    : {len(X_raw):,}")
print(f"Total subjects   : {len(np.unique(subjects))}"
      f" (9 PAMAP2 + 15 PPG-DaLiA)")
print(f"HR range         : {hr_raw.min():.0f} – "
      f"{hr_raw.max():.0f} BPM")
print(f"HR mean          : {hr_raw.mean():.1f} BPM")
print(f"Activities       : rest, walk, run, intense")
print(f"Features         : {len(feature_names)}")
print(f"\nFeature correlations with HR Z-Score:")
df_corr = pd.DataFrame(X_scaled, columns=feature_names)
corr_hr = df_corr.corr()['HR Z-Score'].drop('HR Z-Score')
for feat, val in corr_hr.sort_values(ascending=False).items():
    print(f"  {feat:25s}: {val:+.3f}")

print("\nSaved plots:")
for i in range(1, 9):
    print(f"  eda_0{i}_*.png")
print("\nEDA complete ✓")