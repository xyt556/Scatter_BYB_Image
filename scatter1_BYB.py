"""
Otsu thresholding
==================

This example illustrates automatic Otsu thresholding.
"""

from __future__ import print_function
import numpy as np
from skimage import data
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure
from scipy import optimize
import numpy as np
import gdal
#import pandas as pd


def matrix_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def f_1(x, A, B):
    return A*x + B

ds1 = gdal.Open("./S2A.dat")
ds2 = gdal.Open("./LC8.dat")
camera1 = np.array(ds1.GetRasterBand(2).ReadAsArray())

#camera11 = camera.flatten()
camera11 = camera1.flatten()
camera11 = [i for i in camera11 if (i>=-1 and i<=1)]
camera11 = np.asarray(camera11)

camera2 = np.array(ds2.GetRasterBand(2).ReadAsArray())
camera21 = camera2.flatten()
camera21 = [i for i in camera21 if (i>=-1 and i<=1)]
camera21 = np.asarray(camera21)
#camera = data.camera()
#Add Const
camera11_1=sm.add_constant(camera11)
model = sm.OLS(camera21, camera11_1)
results = model.fit()

print(results.summary())

print('Parameters: ', results.params)
print('R2: ', results.rsquared)


pa = np.asarray (results.params)

print (pa[0])

y_fitted = results.fittedvalues
fig, ax = plt.subplots(figsize=(8,8))
#ax.plot(camera11, camera21, 'o', label='data')
plt.scatter(camera11,camera21,s= 0.2)
ax.plot(camera11, y_fitted, 'r--.',label='OLS')


ftext = 'y =  {:.3f} + {:.3f}x'.format(pa[0],pa[1],)
rtext = 'R-squared= ' + str(results.rsquared)[0:4]


plt.xlim(0, 0.4)
plt.ylim(0, 0.4)

plt.figtext(.15,.8, ftext, fontsize=11, ha='left')
plt.figtext(.15,.77, rtext, fontsize=11, ha='left')
plt.figtext(0.7,.1, "Green", fontsize=12, ha='left')
plt.ylabel('Landsat-8 OLI/TIRS')
plt.xlabel('Sentinel-2 MSI')

plt.tight_layout()
plt.show()
