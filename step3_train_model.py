import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# ── Set working directory ─────────────────────────────────────
os.chdir(r'D:\CTAIML FINAL WORK')

# ── Load features ─────────────────────────────────────────────
X_scaled  = np.load('X_scaled.npy')
subjects  = np.load('subjects.npy')
hr_raw    = np.load('hr_raw.npy')

print(f"Loaded  : {X_scaled.shape[0]:,} windows x {X_scaled.shape[1]} features")
print(f"Subjects: {len(np.unique(subjects))}")

np.random.seed(42)
tf.random.set_seed(42)

# ── Split — keep subjects together (no data leakage) ──────────
unique_subs = np.unique(subjects)
np.random.shuffle(unique_subs)

test_subs  = unique_subs[:4]
train_subs = unique_subs[4:]

train_mask = np.isin(subjects, train_subs)
test_mask  = np.isin(subjects, test_subs)

X_train = X_scaled[train_mask]
X_test  = X_scaled[test_mask]

print(f"\nTrain : {len(X_train):,} windows ({len(train_subs)} subjects)")
print(f"Test  : {len(X_test):,} windows ({len(test_subs)} subjects — never seen)")

X_tr, X_val = train_test_split(X_train, test_size=0.15, random_state=42)
print(f"Train/Val split: {len(X_tr):,} / {len(X_val):,}")

# ── Build autoencoder ─────────────────────────────────────────
def build_autoencoder(input_dim=8):
    inputs = tf.keras.Input(shape=(input_dim,), name='input')

    # Encoder
    x          = tf.keras.layers.Dense(16, activation='relu', name='enc1')(inputs)
    x          = tf.keras.layers.Dense(8,  activation='relu', name='enc2')(x)
    bottleneck = tf.keras.layers.Dense(4,  activation='relu', name='bottleneck')(x)

    # Decoder
    x       = tf.keras.layers.Dense(8,        activation='relu',   name='dec1')(bottleneck)
    x       = tf.keras.layers.Dense(16,       activation='relu',   name='dec2')(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', name='output')(x)

    model = tf.keras.Model(inputs, outputs, name='HeartAE')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    return model

model = build_autoencoder()
model.summary()

total_params = model.count_params()
print(f"\nTotal parameters : {total_params}")
print(f"Est. model size  : ~{total_params // 1024 + 8} KB after INT8 quantization")

# ── Train ─────────────────────────────────────────────────────
print("\nTraining autoencoder...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        verbose=1
    )
]

history = model.fit(
    X_tr, X_tr,
    validation_data=(X_val, X_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete ✓")

# ── Reconstruction error ──────────────────────────────────────
def recon_error(X):
    pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - pred), axis=1)

train_errors = recon_error(X_tr)
val_errors   = recon_error(X_val)
test_errors  = recon_error(X_test)

print(f"\nReconstruction error (MSE):")
print(f"  Train mean : {train_errors.mean():.4f} +/- {train_errors.std():.4f}")
print(f"  Val mean   : {val_errors.mean():.4f}   +/- {val_errors.std():.4f}")
print(f"  Test mean  : {test_errors.mean():.4f}   +/- {test_errors.std():.4f}")

# ── Anomaly threshold ─────────────────────────────────────────
threshold_95 = np.percentile(train_errors, 95)
threshold_99 = np.percentile(train_errors, 99)

print(f"\nAnomaly thresholds:")
print(f"  95th percentile: {threshold_95:.4f}")
print(f"  99th percentile: {threshold_99:.4f}")

flagged_95 = np.sum(test_errors > threshold_95)
flagged_99 = np.sum(test_errors > threshold_99)
print(f"\nTest windows flagged as anomaly:")
print(f"  At 95th threshold: {flagged_95}/{len(test_errors)} "
      f"({100*flagged_95/len(test_errors):.1f}%)")
print(f"  At 99th threshold: {flagged_99}/{len(test_errors)} "
      f"({100*flagged_99/len(test_errors):.1f}%)")

# ── Synthetic anomaly evaluation ──────────────────────────────
print("\nGenerating synthetic anomalies for evaluation...")
X_anomaly = X_test.copy()
X_anomaly[:, 0] += 3.5   # push hr_zscore up — tachycardia
X_anomaly[:, 7] += 3.0   # push activity_hr_ratio up

anomaly_errors = recon_error(X_anomaly)

print(f"  Normal test error  : {test_errors.mean():.4f}")
print(f"  Anomaly test error : {anomaly_errors.mean():.4f}")
print(f"  Separation ratio   : {anomaly_errors.mean()/test_errors.mean():.1f}x")

y_true   = np.concatenate([np.zeros(len(test_errors)),
                            np.ones(len(anomaly_errors))])
y_scores = np.concatenate([test_errors, anomaly_errors])
roc_auc  = roc_auc_score(y_true, y_scores)

print(f"\nROC-AUC score: {roc_auc:.3f}")
print(f"  0.5 = random | 0.8+ = good | 0.9+ = excellent")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history.history['loss'],     label='train loss')
axes[0].plot(history.history['val_loss'], label='val loss')
axes[0].set_title('Training loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].legend()

axes[1].hist(train_errors,   bins=50, alpha=0.6,
             color='steelblue', label='train normal')
axes[1].hist(test_errors,    bins=50, alpha=0.6,
             color='green',     label='test normal')
axes[1].hist(anomaly_errors, bins=50, alpha=0.6,
             color='red',       label='synthetic anomaly')
axes[1].axvline(threshold_95, color='orange', linewidth=2,
                linestyle='--', label=f'threshold={threshold_95:.3f}')
axes[1].set_title('Reconstruction error distribution')
axes[1].set_xlabel('MSE')
axes[1].legend(fontsize=8)

test_sub_ids = subjects[test_mask]
axes[2].set_title('Test error per held-out subject')
for sid in np.unique(test_sub_ids):
    errs = test_errors[test_sub_ids == sid]
    axes[2].scatter([sid]*len(errs), errs, alpha=0.4, s=10)
axes[2].axhline(threshold_95, color='orange',
                linestyle='--', label='threshold')
axes[2].set_xlabel('Subject ID')
axes[2].set_ylabel('Reconstruction error')
axes[2].legend()

plt.tight_layout()
plt.savefig('training_results.png', dpi=100)
plt.show()

# ── Save ──────────────────────────────────────────────────────
model.save('autoencoder_model.keras')
np.save('threshold_95.npy', np.array([threshold_95]))
np.save('threshold_99.npy', np.array([threshold_99]))

print("\nSaved:")
print("  autoencoder_model.keras")
print("  threshold_95.npy")
print("  threshold_99.npy")
print("\nStep 3 complete — ready for Step 4: TFLite conversion")