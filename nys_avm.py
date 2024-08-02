import polars as pl
import xgboost as xgb
import glob
import os

resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nys_data/*.CSV'))

dats = [pl.read_csv(resi_fn, truncate_ragged_lines=True, infer_schema=False, encoding='iso-8859-1') for resi_fn in resi_fns]
# NB We prophylactically installed fsspec per https://github.com/pola-rs/polars/issues/7714
# Not sure if it made a difference, did not test without it

dats = pl.concat(dats, how="vertical_relaxed")

gis_fn = '~/Documents/Github/dcavm/nys_data/NYS_Address_Points_2452783919555667456.csv'
gis = pl.read_csv(gis_fn,infer_schema_length=10000000)

dats = dats.with_columns(
    pl.concat_str(
        [pl.col('county_name'),pl.col('street_nbr'),pl.col('street_name'),],
        separator=' ',
        ).str.to_lowercase().alias('joinkey'),
    )
gis = gis.with_columns(
    pl.concat_str(
        [pl.col('CountyName'),pl.col('AddressNumber'),pl.col('StreetName'),pl.col('PostType'),],
    separator=' ',
    ).str.to_lowercase().alias('joinkey'),
    )

dats = dats.join(gis,on='joinkey')

