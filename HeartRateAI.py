import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')  # needed for PyCharm plots
import matplotlib.pyplot as plt

# ─── Your exact paths from the screenshots ───────────────────
PAMAP2_PATH = r'D:\CTAIML FINAL WORK\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\Protocol'
PPG_PATH    = r'D:\CTAIML FINAL WORK\ppg+dalia\data\PPG_FieldStudy'

# ─── PAMAP2 column definitions ────────────────────────────────
PAMAP2_COLUMNS = [
    'timestamp', 'activity_id', 'heart_rate',
    'hand_temp',
    'hand_acc_x',  'hand_acc_y',  'hand_acc_z',
    'hand_acc2_x', 'hand_acc2_y', 'hand_acc2_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'hand_mag_x',  'hand_mag_y',  'hand_mag_z',
    'hand_orient_1','hand_orient_2','hand_orient_3','hand_orient_4',
    'chest_temp',
    'chest_acc_x',  'chest_acc_y',  'chest_acc_z',
    'chest_acc2_x', 'chest_acc2_y', 'chest_acc2_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'chest_mag_x',  'chest_mag_y',  'chest_mag_z',
    'chest_orient_1','chest_orient_2','chest_orient_3','chest_orient_4',
    'ankle_temp',
    'ankle_acc_x',  'ankle_acc_y',  'ankle_acc_z',
    'ankle_acc2_x', 'ankle_acc2_y', 'ankle_acc2_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x',  'ankle_mag_y',  'ankle_mag_z',
    'ankle_orient_1','ankle_orient_2','ankle_orient_3','ankle_orient_4',
]

ACTIVITY_MAP = {
    1:'rest',  2:'rest',    3:'rest',
    4:'walk',  7:'walk',   12:'walk',  13:'walk',
    5:'run',  24:'run',    20:'run',
    6:'intense', 16:'intense', 17:'intense'
}

# ─── Load PAMAP2 ──────────────────────────────────────────────
def load_pamap2():
    all_subjects = []

    for i in range(101, 110):
        fpath = os.path.join(PAMAP2_PATH, f'subject{i}.dat')
        if not os.path.exists(fpath):
            print(f'  subject{i} not found, skipping')
            continue

        df = pd.read_csv(fpath, sep=' ', header=None,
                         names=PAMAP2_COLUMNS)
        df['subject_id'] = i
        df['activity']   = df['activity_id'].map(ACTIVITY_MAP)

        # Remove transitions and missing HR
        df = df[df['activity_id'] != 0]
        df = df.dropna(subset=['heart_rate', 'activity'])

        # Keep only columns we need
        df = df[['timestamp','subject_id','activity',
                 'heart_rate',
                 'hand_acc_x','hand_acc_y','hand_acc_z',
                 'hand_gyro_x','hand_gyro_y','hand_gyro_z']]

        all_subjects.append(df)
        print(f'  subject{i} → {len(df):,} rows | '
              f'HR: {df.heart_rate.min():.0f}–'
              f'{df.heart_rate.max():.0f} BPM | '
              f'activities: {df.activity.unique()}')

    return pd.concat(all_subjects, ignore_index=True)



# ─── Load PPG-DaLiA ───────────────────────────────────────────
def load_ppg_dalia():
    all_subjects = []

    PPG_ACTIVITY_MAP = {
        1:'rest', 2:'walk', 3:'intense',
        4:'intense', 5:'rest', 6:'rest', 7:'walk'
    }

    # Sampling rates from PPG-DaLiA readme
    ACC_FS   = 32    # wrist ACC at 32 Hz
    LABEL_FS = 0.125 # one HR label every 8 seconds
    WIN_SAMPLES = ACC_FS * 8  # 256 samples per 8-sec window

    for sid in range(1, 16):
        fpath = os.path.join(PPG_PATH, f'S{sid}', f'S{sid}.pkl')
        if not os.path.exists(fpath):
            print(f'  S{sid} not found, skipping')
            continue

        with open(fpath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Extract wrist accelerometer — shape (N, 3)
        acc = data['signal']['wrist']['ACC']

        # Ground truth HR from ECG — shape (M,) one per 8 sec
        hr_labels = data['label']

        # Activity labels — shape (N, 1) at full resolution, downsample to match HR labels
        activity_full = data['activity'].flatten()

        # Downsample activity to one label per 8-sec window
        # activity is at same rate as ACC (32 Hz), so 256 samples per window
        activity_windows = []
        acc_mean_list = []
        acc_std_list  = []

        for w in range(len(hr_labels)):
            start = w * WIN_SAMPLES
            end   = start + WIN_SAMPLES

            if end > len(acc):
                break

            # Accelerometer stats for this window
            chunk = acc[start:end]
            mag   = np.sqrt(chunk[:,0]**2 + chunk[:,1]**2 + chunk[:,2]**2)
            acc_mean_list.append(np.mean(mag))
            acc_std_list.append(np.std(mag))

            # Dominant activity in this window
            act_chunk = activity_full[start:end]
            # Most frequent non-zero activity
            vals, counts = np.unique(act_chunk[act_chunk > 0], return_counts=True)
            if len(vals) > 0:
                activity_windows.append(int(vals[np.argmax(counts)]))
            else:
                activity_windows.append(0)

        # Trim all arrays to same length
        n = min(len(hr_labels), len(activity_windows))

        df = pd.DataFrame({
            'subject_id'   : sid,
            'heart_rate'   : hr_labels[:n],
            'activity_code': activity_windows[:n],
            'a_mean'       : acc_mean_list[:n],
            'a_std'        : acc_std_list[:n],
        })

        df['activity'] = df['activity_code'].map(PPG_ACTIVITY_MAP)

        # Clean up
        df = df.dropna(subset=['heart_rate', 'activity', 'a_mean'])
        df = df[df['heart_rate'] > 30]   # remove impossible HR values
        df = df[df['heart_rate'] < 220]
        df = df[df['activity_code'] > 0] # remove unlabelled windows

        all_subjects.append(df)
        print(f'  S{sid:>2} → {len(df):>4} windows | '
              f'HR: {df.heart_rate.min():.0f}–{df.heart_rate.max():.0f} BPM | '
              f'activities: {df.activity.unique()}')

    return pd.concat(all_subjects, ignore_index=True)


# ─── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('Loading PAMAP2...')
    print('=' * 55)
    pamap2_df = load_pamap2()

    print(f'\nPAMAP2 total : {len(pamap2_df):,} rows')
    print(f'Subjects     : {pamap2_df.subject_id.nunique()}')
    print(f'HR range     : {pamap2_df.heart_rate.min():.0f}–'
          f'{pamap2_df.heart_rate.max():.0f} BPM')

    print('\n' + '=' * 55)
    print('Loading PPG-DaLiA...')
    print('=' * 55)
    ppg_df = load_ppg_dalia()

    print(f'\nPPG-DaLiA total : {len(ppg_df):,} windows')
    print(f'Subjects        : {ppg_df.subject_id.nunique()}')
    print(f'HR range        : {ppg_df.heart_rate.min():.0f}–'
          f'{ppg_df.heart_rate.max():.0f} BPM')

    # ── Quick plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    pamap2_df.boxplot(column='heart_rate',
                      by='activity', ax=axes[0])
    axes[0].set_title('PAMAP2 — HR by activity')
    axes[0].set_xlabel('Activity')
    axes[0].set_ylabel('HR (BPM)')

    ppg_df.boxplot(column='heart_rate',
                   by='activity', ax=axes[1])
    axes[1].set_title('PPG-DaLiA — HR by activity')
    axes[1].set_xlabel('Activity')
    axes[1].set_ylabel('HR (BPM)')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=100)
    plt.show()

    print('\nBoth datasets loaded successfully ✓')
    print('Plot saved as data_overview.png')

    # ── Save for next step ────────────────────────────────────
    pamap2_df.to_pickle('pamap2_cleaned.pkl')
    ppg_df.to_pickle('ppg_dalia_cleaned.pkl')
    print('Cleaned data saved ✓')
    print('  pamap2_cleaned.pkl')
    print('  ppg_dalia_cleaned.pkl')