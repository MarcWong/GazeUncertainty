import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="white",context="talk")

# Set up the matplotlib figure
f, (ax) = plt.subplots(1, 1, figsize=(12, 7), dpi=144)

# Generate some sequential data
y1 = np.array([.55, .61, .66, .70, .74, .77, .81, .847, .88,
               .905, .925, .94, .951, .96])
bar1 = np.array([.557, .623, .682, .729, .771, .804, .835, .879, .911,
                .935, .951, .961, .970, .978])
pie1 = np.array([.64, .67, .71, .74, .77, .79, .83, .85, .88,
                .896, .914, .93, .94, .947])
scatter1 = np.array([.46, .52, .58, .63, .68, .72, .77, .82, .86,
                    .895, .92, .941, .954, .964])
line1 = np.array([.41, .46, .51, .55, .59, .63, .67, .71, .76,
                 .795, .828, .853, .875, .893])
x1 = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0])

data=pd.DataFrame(index=x1, data={'average':y1, 'bar': bar1, 'pie': pie1, 'scatter': scatter1, 'line': line1})

sns.lineplot(data=data, ax=ax, markers=True, dashes=False)
ax.set(ylim=(0.3, 1.02))
#ax.set(xlim=(0, 10))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_xlabel("AOI Enlargement (°)")
ax.set_ylabel("Hit Any AOI Rate")

ax.axhline(y=0.4, color='#BBBBBB', linestyle='--')
ax.axhline(y=0.5, color='#BBBBBB', linestyle='--')
ax.axhline(y=0.6, color='#BBBBBB', linestyle='--')
ax.axhline(y=0.7, color='#BBBBBB', linestyle='--')
ax.axhline(y=0.8, color='#BBBBBB', linestyle='--')
ax.axhline(y=0.9, color='#BBBBBB', linestyle='--')
ax.axhline(y=1.0, color='#BBBBBB', linestyle='--')

# Finalize the plot
sns.despine()
plt.setp(f.axes, yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#plt.tight_layout(h_pad=2)

plt.show()
#f.savefig('FixationCoverage.pdf')