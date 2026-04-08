import matplotlib.pyplot as plt
import numpy as np

lb_percent = [0, 10, 25, 50, 75, 100]
absorbance = [0.000, 0.508, 1.149, 1.451, 1.523, 1.577]

k0 = 1.577
k_over_k0 = [a / k0 for a in absorbance]

# Split into two groups
x1, y1 = lb_percent[:3], k_over_k0[:3]   # first three points:  0, 10, 25%
x2, y2 = lb_percent[3:], k_over_k0[3:]   # second three points: 50, 75, 100%

# Linear fits
m1, b1 = np.polyfit(x1, y1, 1)
m2, b2 = np.polyfit(x2, y2, 1)

# Intersection: m1*x + b1 = m2*x + b2
x_int = (b2 - b1) / (m1 - m2)
y_int = m1 * x_int + b1

# Smooth lines for plotting
x1_line = np.linspace(-5, x_int, 200)
x2_line = np.linspace(x_int, 105, 200)

fig, ax = plt.subplots(figsize=(8, 5))

# Data points
ax.scatter(lb_percent, k_over_k0, color='steelblue', zorder=5, s=50)

# Lines of best fit
ax.plot(x1_line, m1 * x1_line + b1, color='tomato', linewidth=2,
        label=f'Fit 1: y = {m1:.4f}x + {b1:.4f}')
ax.plot(x2_line, m2 * x2_line + b2, color='seagreen', linewidth=2,
        label=f'Fit 2: y = {m2:.4f}x + {b2:.4f}')

# Intersection point
ax.scatter(x_int, y_int, color='black', zorder=6, s=80)
ax.annotate(f'Threshold\n({x_int:.1f}%, {y_int:.3f})',
            xy=(x_int, y_int),
            xytext=(x_int + 8, y_int - 0.12),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', lw=0.8))

ax.set_xlabel('% LB', fontsize=13)
ax.set_ylabel('k / k₀', fontsize=13)
ax.set_xlim(-5, 105)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('spectrophotometer_plot.png', dpi=150)
plt.show()
