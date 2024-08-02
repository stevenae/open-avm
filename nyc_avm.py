import polars as pl
import xgboost as xgb
import glob
import os

resi_fn = '~/Documents/Github/dcavm/nyc_data/rollingsales_manhattan.xlsx'
resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

# resi_stub = os.path.expanduser('~/Documents/Github/dcavm/nyc_data/20')
# first_fns = [glob.glob(resi_stub+suf) for suf in ['0*','10*']]
# first_fns = [glob.glob(resi_stub+suf) for suf in ['0*','10*']]

dats = []
for resi_fn in resi_fns:
    dat = pl.read_excel(resi_fn,engine="calamine")
    header_row_ind = dat.select(pl.first().str.contains('BOROUGH')).to_numpy().nonzero()[0][0]
    dat = dat.rename(dat[header_row_ind,].to_dicts().pop())
    dat = dat.rename(str.strip)
    dat = dat.slice(header_row_ind+1)
    dats.append(dat)

dats = pl.concat(dats, how="diagonal_relaxed")

dats = dats.filter(pl.col('TAX CLASS AT TIME OF SALE').str.contains_any(['1','2']))

dats = dats.with_columns(block_pad = pl.col('BLOCK').cast(pl.String).str.zfill(5))
dats = dats.with_columns(lot_pad = pl.col('LOT').cast(pl.String).str.zfill(4))
dats = dats.with_columns(
    pl.concat_str(
        [
            pl.col("BOROUGH"),
            pl.col("block_pad"),
            pl.col("lot_pad"),
        ],
        separator="",
    ).cast(pl.Float64).alias("bbl"),
)

gis_fn = '~/Documents/Github/dcavm/nyc_data/pluto_24v2.csv'
gis = pl.read_csv(gis_fn,infer_schema_length=10000)
# dats_anti = dats.join(gis,on='bbl',how='anti')
# dats["BOROUGH"].value_counts().join(
#     dats_anti["BOROUGH"].value_counts(), on="BOROUGH"
# ).sort("BOROUGH")
# gis.filter((pl.col('borocode') == 5) & (pl.col('block') == 5742) & (pl.col('lot') == 1008))

dats = dats.join(gis,on='bbl')

dats.columns

# TODO: make relative (e.g., 5th - 95th percentile of sales prices)
dats = dats.with_columns(pl.col('SALE PRICE').cast(pl.Float64))
dats = dats.filter(pl.col('SALE PRICE')>1e5)
dats = dats.filter(pl.col('SALE PRICE')<2e6)

dats = dats.with_columns(
    pl.col('yearbuilt').replace(0,None),
    pl.col('yearalter1').replace(0,None),
    pl.col('yearalter2').replace(0,None)
)

categorical_name_array = ['landuse','ltdheight','bldgclass']
numerical_col_name_array = ['yearbuilt','yearalter1','yearalter2']
price_col_name = 'SALE PRICE'
date_col_name = 'SALE DATE'
dats = data_prep_for_model(dats,date_col_name,price_col_name,numerical_col_name_array,categorical_name_array)
fmt_str = "%Y-%m-%d %H:%M:%S"
dats = date_format(dats,date_col_name,fmt_str)
xgb_data = dats
xgb_data[date_col_name]
fit_model(xgb_data,date_col_name,price_col_name)
