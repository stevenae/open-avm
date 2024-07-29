import polars as pl
import xgboost as xgb
import glob
import os

resi_fn = '~/Documents/Github/dcavm/nyc_data/rollingsales_manhattan.xlsx'
resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

dats = [pl.read_excel(resi_fn,engine="calamine",read_csv_options={ "header_row": 4}) for resi_fn in resi_fns]
pl.concat([dats[0],dats[1],dats[2],dats[3],dats[4]], how="vertical_relaxed")
pl.concat(dats, how="vertical_relaxed")

# dat['SALEVAL_DESC'].value_counts().sort(by='count').tail(10)
dat = dat.filter(pl.col('SALEVAL_DESC')=='Valid and verified sale')
