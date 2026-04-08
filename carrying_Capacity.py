import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── ACS-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.5,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.6,
    'ytick.minor.width':  0.6,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'xtick.minor.size':   2.0,
    'ytick.minor.size':   2.0,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'lines.linewidth':    1.2,
    'figure.dpi':         300,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'axes.spines.top':    True,
    'axes.spines.right':  True,
})

# ── Data ────────────────────────────────────────────────────────────────────
lb_percent = np.array([0, 10, 25, 50, 75, 100])
absorbance  = np.array([0.000, 0.508, 1.149, 1.451, 1.523, 1.577])

k0        = 1.577
k_over_k0 = absorbance / k0

# ── Linear fits ─────────────────────────────────────────────────────────────
x1, y1 = lb_percent[:3], k_over_k0[:3]
x2, y2 = lb_percent[3:], k_over_k0[3:]

m1, b1 = np.polyfit(x1, y1, 1)
m2, b2 = np.polyfit(x2, y2, 1)

x_int = (b2 - b1) / (m1 - m2)
y_int =  m1 * x_int + b1

# R² values
def r_squared(x, y, m, b):
    y_pred = m * np.array(x) + b
    ss_res = np.sum((np.array(y) - y_pred) ** 2)
    ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

r2_1 = r_squared(x1, y1, m1, b1)
r2_2 = r_squared(x2, y2, m2, b2)

# Lines
x1_line = np.linspace(-3, x_int, 300)
x2_line = np.linspace(x_int, 103, 300)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.2))

# Grid (subtle)
ax.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', zorder=0)

# Lines of best fit
ax.plot(x1_line, m1 * x1_line + b1,
        color='#C0392B', linewidth=1.2, zorder=3,
        label=fr'$y = {m1:.4f}x {b1:+.4f}$  ($R^2={r2_1:.4f}$)')
ax.plot(x2_line, m2 * x2_line + b2,
        color='#27AE60', linewidth=1.2, zorder=3,
        label=fr'$y = {m2:.4f}x {b2:+.4f}$  ($R^2={r2_2:.4f}$)')

# Data points
ax.scatter(lb_percent, k_over_k0,
           color='#2C3E50', s=18,
           edgecolors='white', linewidths=0.5,
           zorder=5, label='Data')

# Intersection marker
ax.scatter(x_int, y_int, color='black', zorder=6,
           s=25, marker='D', linewidths=0.5)

ax.annotate(
    f'Threshold\n({x_int:.1f}%, {y_int:.3f})',
    xy=(x_int, y_int),
    xytext=(x_int + 10, y_int - 0.13),
    fontsize=7,
    arrowprops=dict(arrowstyle='->', color='black',
                    lw=0.8, shrinkA=3, shrinkB=3),
    bbox=dict(boxstyle='round,pad=0.25', fc='#FFFDE7',
              ec='#999999', lw=0.6),
)

# Minor ticks on all four sides
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.tick_params(which='minor', direction='in', top=True, right=True, length=2, width=0.5)
ax.tick_params(which='major', direction='in', top=True, right=True)

ax.set_xlabel('LB content (%)', labelpad=4)
ax.set_ylabel(r'$k\,/\,k_0$', labelpad=4)
ax.set_xlim(-5, 105)
ax.set_ylim(-0.04, 1.10)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0', '25', '50', '75', '100'])

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
          frameon=True, framealpha=0.9, edgecolor='#cccccc',
          borderpad=0.5, labelspacing=0.3,
          handlelength=1.6, handletextpad=0.4)

plt.tight_layout(pad=0.4)
plt.subplots_adjust(right=0.62)
plt.savefig('spectrophotometer_plot.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('spectrophotometer_plot.pdf', bbox_inches='tight', facecolor='white')
plt.show()
