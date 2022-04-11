import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

import matplotlib as mpl
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


#
# Plot the beta distribution
#
params = [ (2,4,'r'), (2,2, 'g'), (4,2, 'b')]
fig = plt.figure(figsize=(10,5))
ax = fig.subplots()
plt.xlim(0, 1)
for a,b,color in params:
    x = np.linspace(beta.ppf(0, a, b),beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), '{}'.format(color), label = 'alpha={},beta={}'.format(a,b))
ax.legend()
# ax.tick_params(axis='x', colors='w')  
ax.tick_params(axis='y', colors='w')
#change tick to R
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[-1] = 'R'
labels[0] = '0'
ax.set_xticklabels(labels)

plt.title('Parameter r distribution', fontsize='15')
plt.xlabel('r', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.savefig('beta.jpg')