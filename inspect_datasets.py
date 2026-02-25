"""
ECG Dataset Inspector
Inspects zheng and ptb TensorFlow datasets - size, signal length, and plots.

Written by Cladue
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

DATA_DIR = '/data/wadh6616/VAE_ECG_data/mean_beat/'

datasets = {
    'zheng': None,
    'ptb':   None,
}

# ─── HELPER: flatten nested dict ───────────────────────────────────────────────
def print_structure(d, prefix='  ', depth=0):
    if depth > 3:
        return
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, dict):
                print(f"{prefix}'{key}': (nested dict)")
                print_structure(val, prefix + '  ', depth + 1)
            elif hasattr(val, 'shape'):
                print(f"{prefix}'{key}': shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"{prefix}'{key}': {type(val)}")
    elif hasattr(d, 'shape'):
        print(f"{prefix}shape={d.shape}, dtype={d.dtype}")

def find_ecg_signal(sample):
    """Try to find the main ECG signal in the sample."""
    candidates = ['ecg', 'beat', 'signal', 'data', 'waveform']
    if isinstance(sample, dict):
        for key in candidates:
            if key in sample:
                val = sample[key]
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        if hasattr(v2, 'numpy') and len(v2.shape) >= 1:
                            return key, k2, v2.numpy()
                elif hasattr(val, 'numpy') and len(val.shape) >= 1:
                    return key, None, val.numpy()
        # Fallback: find largest 1D+ tensor
        for key, val in sample.items():
            if hasattr(val, 'numpy') and len(val.shape) >= 1 and val.shape[0] > 10:
                return key, None, val.numpy()
    return None, None, None

# ─── LOAD DATASETS ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading datasets...")
print("=" * 60)

for name in datasets:
    try:
        ds, info = tfds.load(name, split='train', data_dir=DATA_DIR,
                             with_info=True, shuffle_files=False)
        datasets[name] = (ds, info)
        print(f"\n✓ Loaded: {name}")
    except Exception as e:
        print(f"\n✗ Failed to load {name}: {e}")

# ─── INSPECT EACH DATASET ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

signal_samples = {}

for name, data in datasets.items():
    if data is None:
        print(f"\n{name}: Failed to load, skipping.")
        continue

    ds, info = data

    # Count entries
    num_entries = ds.cardinality().numpy()
    if num_entries < 0:
        print(f"\n{name}: Counting entries (may take a moment)...")
        num_entries = sum(1 for _ in ds)

    print(f"\n{'─'*40}")
    print(f"Dataset : {name.upper()}")
    print(f"Entries : {num_entries:,}")

    # Get one sample
    sample = next(iter(ds.skip(100)))  # Change the number to decide which you want to plot. 
    # Prevents issues like plotting tachycardia. 

    print(f"\nFields:")
    print_structure(sample)

    # Find signal
    key, subkey, sig = find_ecg_signal(sample)
    if sig is not None:
        label = f"{key}.{subkey}" if subkey else key
        print(f"\nECG signal field : '{label}'")
        print(f"Signal shape     : {sig.shape}")
        print(f"Signal length    : {sig.shape[-1]} samples")
        if len(sig.shape) > 1:
            print(f"Number of leads  : {sig.shape[0]}")
        print(f"Value range      : [{sig.min():.4f}, {sig.max():.4f}]")
        signal_samples[name] = (label, sig)
    else:
        print(f"\n⚠ Could not automatically find ECG signal field")

# ─── PLOT SIGNALS ──────────────────────────────────────────────────────────────
if signal_samples:
    print("\n" + "=" * 60)
    print("Plotting sample signals...")
    print("=" * 60)

    n_datasets = len(signal_samples)
    fig = plt.figure(figsize=(16, 5 * n_datasets))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle('ECG Dataset — Sample Signals', fontsize=16,
                 color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(n_datasets, 1, hspace=0.6)
    colors = {'zheng': '#00d4ff', 'ptb': '#ff6b6b'}

    for idx, (name, (label, sig)) in enumerate(signal_samples.items()):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor('#1a1a2e')

        # Handle multi-lead — plot first lead
        if len(sig.shape) > 1:
            signal_to_plot = sig[0]
            lead_info = f" | Lead 1/{sig.shape[0]}"
        else:
            signal_to_plot = sig
            lead_info = ""

        x = np.arange(len(signal_to_plot))
        color = colors.get(name, '#ffffff')
        ax.plot(x, signal_to_plot, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(x, signal_to_plot, alpha=0.08, color=color)

        ax.set_title(
            f"{name.upper()}  |  field: '{label}'{lead_info}  |  {len(signal_to_plot)} samples",
            color='white', fontsize=11, pad=8)
        ax.set_xlabel('Sample index', color='#aaaaaa', fontsize=9)
        ax.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#aaaaaa')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.15, color='#ffffff')

    save_path = '/users/wadh6616/VAE_ECG/dataset_inspection/dataset_inspection.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"\n✓ Plot saved to: {save_path}")

print("\nDone!")