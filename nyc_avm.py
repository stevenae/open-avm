import polars as pl
import xgboost as xgb
import glob
import os

resi_fn = '~/Documents/Github/dcavm/nyc_data/rollingsales_manhattan.xlsx'
resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

dats = [pl.read_excel(resi_fn,engine="calamine",read_options={ "header_row": 4}) for resi_fn in resi_fns]
dats = pl.concat(dats, how="vertical_relaxed")

dats = dats.with_columns(
    pl.concat_str(
        [
            pl.col("BOROUGH"),
            pl.col("BLOCK"),
            pl.col("LOT"),
        ],
        separator="",
    ).cast(pl.Int64).alias("bbl"),
)

gis_fn = '~/Documents/Github/dcavm/nyc_data/Primary_Land_Use_Tax_Lot_Output__PLUTO_.csv'
gis = pl.read_csv(gis_fn,infer_schema_length=10000)

dats = dats.join(gis,on='bbl')
