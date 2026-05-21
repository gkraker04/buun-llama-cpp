#!/usr/bin/env python3
"""Generate benchmark chart from sweep results CSV."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

csv_path = sys.argv[1] if len(sys.argv) > 1 else "G:/hermes/buun-llama-cpp/experiments/speed/results/sweep_results.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "G:/hermes/buun-llama-cpp/experiments/speed/results/sweep_chart.png"

rows = []
with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r['status'] == 'ok':
            rows.append({
                'batch': int(r['batch_size']),
                'ubatch': int(r['ubatch_size']),
                'decode': float(r['decode_tok_s']),
                'prompt': float(r['prompt_tok_s']),
            })

if not rows:
    print("No valid results found")
    sys.exit(1)

# Sort by batch then ubatch
rows.sort(key=lambda r: (r['batch'], r['ubatch']))

# Group by batch
batches = sorted(set(r['batch'] for r in rows))
batch_groups = {b: [r for r in rows if r['batch'] == b] for b in batches}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#1a1a2e')

# Color palette
colors = ['#e94560', '#f5a623', '#0f3460', '#16213e', '#533483']
bar_colors = {
    256: '#e94560',
    512: '#f5a623',
    1024: '#0fb9b1',
    2048: '#a29bfe',
}

bar_width = 0.18
x = np.arange(len(batches))

# --- Decode Chart ---
for i, b in enumerate(batches):
    group = batch_groups[b]
    u_labels = [str(r['ubatch']) for r in group]
    d_vals = [r['decode'] for r in group]
    
    xs = np.arange(len(group)) + i * (len(batches) + 1)
    bars = ax1.bar(xs, d_vals, width=0.8, color=bar_colors[b], edgecolor='white', linewidth=0.5, alpha=0.9, label=f'batch={b}')
    for bar, val in zip(bars, d_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
    # ubatch labels
    for j, xs_pos in enumerate(xs):
        ax1.text(xs_pos, -1.2, str(group[j]['ubatch']), ha='center', va='top', fontsize=6, color='#aaa', rotation=45)

ax1.set_xlabel('Batch Size → (Ubatch labels below bars)', color='white')
ax1.set_ylabel('Decode (tok/s)', color='white')
ax1.set_title('Decode Speed by Batch & Ubatch', color='white', fontsize=14, fontweight='bold')
ax1.set_xticks([])
ax1.tick_params(colors='white')
ax1.set_facecolor('#16213e')
ax1.grid(axis='y', alpha=0.3, color='#555')
ax1.spines['bottom'].set_color('#555')
ax1.spines['left'].set_color('#555')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Prompt Chart ---    
for i, b in enumerate(batches):
    group = batch_groups[b]
    p_vals = [r['prompt'] for r in group]
    
    xs = np.arange(len(group)) + i * (len(batches) + 1)
    ax2.bar(xs, p_vals, width=0.8, color=bar_colors[b], edgecolor='white', linewidth=0.5, alpha=0.9, label=f'batch={b}')
    for j, xs_pos in enumerate(xs):
        ax2.text(xs_pos, -1, str(group[j]['ubatch']), ha='center', va='top', fontsize=6, color='#aaa', rotation=45)

ax2.set_xlabel('Batch Size → (Ubatch labels below bars)', color='white')
ax2.set_ylabel('Prompt (tok/s)', color='white')
ax2.set_title('Prompt Processing Speed by Batch & Ubatch', color='white', fontsize=14, fontweight='bold')
ax2.set_xticks([])
ax2.tick_params(colors='white')
ax2.set_facecolor('#16213e')
ax2.grid(axis='y', alpha=0.3, color='#555')
ax2.spines['bottom'].set_color('#555')
ax2.spines['left'].set_color('#555')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=bar_colors[b]) for b in batches]
labels = [f'batch={b}' for b in batches]
fig.legend(handles, labels, loc='lower center', ncol=4, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Chart saved to {out_path}")
print(f"File size: {os.path.getsize(out_path)} bytes")
