import polars as pl

xgb_err = pl.read_csv('~/Documents/xgb_errors.csv')
xgb_err.select('error').describe()

import matplotlib.pyplot as plt
plt.scatter(xgb_err.select('PRICE'),xgb_err.select('error'))
plt.show()
