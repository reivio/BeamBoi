import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 1, 0.1
s1 = np.random.normal(mu, sigma, 1000)

plt.plot(s1)
count, bins, ignored = plt.hist(s1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.show()