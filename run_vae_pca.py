"""
Run PCA and a small VAE hyperparameter search on the dataset.
Saves: pca_pc1_loadings.png, recon_error.png, roc_vae.png, df_with_latents.csv, best_vae_model.h5
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data (adjust paths if needed)
df_dos = pd.read_csv(r'D:\RawData-20251206T072629Z-3-001\RawData\Dos-Drone\dos1.csv')
df_mal = pd.read_csv(r'D:\RawData-20251206T072629Z-3-001\RawData\Malfunction-Drone\malfunction1.csv')
df_norm = pd.read_csv(r'D:\RawData-20251206T072629Z-3-001\RawData\NormalFlight\normal1.csv')

for d in [df_dos, df_mal, df_norm]:
    if 'label' not in d.columns:
        # labels may be set in notebook; set here just in case
        pass

# set labels
df_norm['label'] = 'Normal'
df_dos['label'] = 'DoS_Attack'
df_mal['label'] = 'Malfunction'

# concat
df = pd.concat([df_dos, df_mal, df_norm], ignore_index=True)
print('Combined shape:', df.shape)

# drop obvious id/time cols
drop_cols = [c for c in df.columns if ("_Time" in c) or (".seq" in c) or (".secs" in c) or (c in ['S.No', 'S.No.'])]
print('Dropping cols (time/seq/id):', len(drop_cols))
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# numeric features
numeric = df.select_dtypes(include=[np.number]).copy()
print('Numeric shape:', numeric.shape)
if numeric.shape[1] == 0:
    raise SystemExit('No numeric features')

# PCA
numeric_fill = numeric.fillna(numeric.median())
scaler = StandardScaler()
Xp = scaler.fit_transform(numeric_fill)
pc = min(10, Xp.shape[1])
pca = PCA(n_components=pc)
pca.fit(Xp)
explained = np.round(pca.explained_variance_ratio_,4)
print('Explained variance ratio:', explained)

# print top contributors for first 3 PCs
for i in range(min(3, pca.components_.shape[0])):
    comp = pca.components_[i]
    idx = np.argsort(np.abs(comp))[::-1][:10]
    feats = numeric.columns[idx].tolist()
    weights = comp[idx]
    print(f"\nPC{i+1} top features:")
    for f,w in zip(feats, weights):
        print(f"  {f}: {w:.4f}")

# plot PC1 loadings
loadings = np.abs(pca.components_[0])
order = np.argsort(loadings)[::-1]
topn = min(20, len(order))
plt.figure(figsize=(12,4))
plt.bar(range(topn), loadings[order][:topn])
plt.xticks(range(topn), numeric.columns[order][:topn], rotation=90)
plt.title('Top absolute loadings for PC1')
plt.tight_layout()
plt.savefig('pca_pc1_loadings.png')
print('Saved pca_pc1_loadings.png')

# --- VAE part ---
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, classification_report

# prepare data
X_all = Xp
labels = df['label'].values
is_normal = (labels == 'Normal')
X_normal = X_all[is_normal]
y_binary = np.where(is_normal, 0, 1)
print('Total rows', X_all.shape[0], 'Normal for training', X_normal.shape[0])

# VAE builder
from tensorflow.keras import backend as K

def build_vae(input_dim, latent_dim=16, enc_layers=[256,128], dec_layers=[128,256], activation='relu', beta=1.0):
    inp = Input(shape=(input_dim,))
    x = inp
    for h in enc_layers:
        if activation == 'leaky_relu':
            x = layers.Dense(h, activation='linear')(x)
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Dense(h, activation=activation)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    def sample_z(args):
        mean, log_var = args
        eps = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(0.5 * log_var) * eps
    z = layers.Lambda(sample_z)([z_mean, z_log_var])
    # decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for h in dec_layers:
        if activation == 'leaky_relu':
            x = layers.Dense(h, activation='linear')(x)
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Dense(h, activation=activation)(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    encoder = Model(inp, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, outputs, name='decoder')
    z_mean_, z_log_var_, z_ = encoder(inp)
    recon = decoder(z_)
    vae = Model(inp, recon)
    # loss
    recon_loss = tf.reduce_mean(tf.square(inp - recon), axis=1)
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var_ - tf.square(z_mean_) - tf.exp(z_log_var_), axis=1)
    vae_loss = tf.reduce_mean(recon_loss + beta * kl_loss)
    vae.add_loss(vae_loss)
    return vae, encoder, decoder

# hyperparam choices (smaller search to save time)
import random
latent_dims = [8,16,32]
enc_options = [[256,128],[512,256,128]]
dec_options = [[128,256],[128,256,512]]
lrs = [1e-3,1e-4]
bs_opts = [32,64]
betas = [0.5,1.0,2.0]
acts = ['relu','elu','leaky_relu']
epochs_options = [50]

n_trials = 4
best_val = np.inf
best_info = None

Xn_train, Xn_val = train_test_split(X_normal, test_size=0.15, random_state=42)

for t in range(n_trials):
    latent = random.choice(latent_dims)
    enc = random.choice(enc_options)
    dec = random.choice(dec_options)
    lr = random.choice(lrs)
    bs = random.choice(bs_opts)
    beta = random.choice(betas)
    act = random.choice(acts)
    epochs = random.choice(epochs_options)
    print(f"Trial {t+1}: latent={latent}, enc={enc}, dec={dec}, lr={lr}, bs={bs}, beta={beta}, act={act}, epochs={epochs}")
    try:
        vae, enc_model, dec_model = build_vae(input_dim=Xn_train.shape[1], latent_dim=latent, enc_layers=enc, dec_layers=dec, activation=act, beta=beta)
    except Exception as e:
        print('build error', e)
        continue
    vae.compile(optimizer=Adam(learning_rate=lr))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = vae.fit(Xn_train, Xn_train, validation_data=(Xn_val, Xn_val), epochs=epochs, batch_size=bs, callbacks=[es], verbose=2)
    val_loss = min(history.history['val_loss'])
    print('val_loss', val_loss)
    if val_loss < best_val:
        best_val = val_loss
        best_info = dict(trial=t+1, latent=latent, enc=enc, dec=dec, lr=lr, bs=bs, beta=beta, act=act, epochs=epochs)
        vae.save('best_vae_model.h5')
        enc_model.save('best_vae_encoder.h5')
        dec_model.save('best_vae_decoder.h5')
        print('saved best models')

print('Best:', best_val, best_info)

# evaluate using saved best model if exists
try:
    best = tf.keras.models.load_model('best_vae_model.h5', compile=False)
    recon_all = best.predict(X_all, batch_size=256)
    recon_mse = np.mean(np.square(X_all - recon_all), axis=1)
    df['recon_mse'] = recon_mse
    y_true = y_binary
    roc = roc_auc_score(y_true, recon_mse)
    ap = average_precision_score(y_true, recon_mse)
    print('ROC AUC:', roc, 'AP:', ap)
    thresh = np.percentile(recon_mse[is_normal], 95)
    print('Threshold (95th pct normal):', thresh)
    y_pred = (recon_mse > thresh).astype(int)
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Normal','Anomaly']))
    # save plots
    plt.figure(figsize=(8,4))
    sns.histplot(df.loc[df['label']=='Normal','recon_mse'], color='green', label='Normal', stat='density', kde=True, alpha=0.6)
    sns.histplot(df.loc[df['label']!='Normal','recon_mse'], color='red', label='Anomalous', stat='density', kde=True, alpha=0.6)
    plt.legend()
    plt.title('Reconstruction error distribution')
    plt.tight_layout()
    plt.savefig('recon_error.png')
    print('Saved recon_error.png')
    fpr, tpr, _ = roc_curve(y_true, recon_mse)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.title('ROC curve')
    plt.tight_layout(); plt.savefig('roc_vae.png')
    print('Saved roc_vae.png')
    # compute latents using encoder if available
    try:
        enc = tf.keras.models.load_model('best_vae_encoder.h5', compile=False)
        z_out = enc.predict(X_all, batch_size=256)
        if isinstance(z_out, list):
            z_mean = z_out[0]
        else:
            # encoder returns single array
            z_mean = z_out
    except Exception:
        try:
            # fallback: use in-session enc_model
            z_mean = enc_model.predict(X_all, batch_size=256)[0]
        except Exception:
            z_mean = None
    if z_mean is not None:
        for i in range(min(z_mean.shape[1], 64)):
            df[f'latent_{i}'] = z_mean[:, i]
        df.to_csv('df_with_latents.csv', index=False)
        print('Saved df_with_latents.csv')
except Exception as e:
    print('Could not evaluate best model:', e)

print('Done')
