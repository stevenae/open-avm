import polars as pl
import xgboost as xgb
import glob
import os

resi_fn = '~/Documents/Github/dcavm/nyc_data/rollingsales_manhattan.xlsx'
resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

dats = [pl.read_excel(resi_fn,engine="calamine",read_options={ "header_row": 4}) for resi_fn in resi_fns]
dats = pl.concat(dats, how="vertical_relaxed")

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
