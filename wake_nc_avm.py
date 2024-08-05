import polars as pl
import xgboost as xgb

resi_fn = '~/Documents/Github/dcavm/wake_nc_data/Parcels.csv'
dat = pl.read_csv(resi_fn,infer_schema_length=100000)

dat = dat.filter(pl.col('TOTSALPRICE')>1e5)
dat = dat.filter(pl.col('TOTSALPRICE')<2e6)

dat = dat.with_columns(
   pl.col("SALE_DATE").str.to_date("%Y/%m/%d %H:%M:%S+00")
)

gis_fn = '~/Documents/Github/dcavm/wake_nc_data/Parcel_Address_Points.csv'
gis = pl.read_csv(gis_fn)

dat = dat.join(gis, on='PIN_NUM')

xgb_data = dat.select(pl.col(['LATITUDE','LONGITUDE',
           'BLDG_VAL','LAND_VAL','TOTAL_VALUE','YEAR_BUILT','DEISGN_STYL',
                        'BEDRM','AYB','YR_RMDL','EYB','STORIES','GBA',
                        'GRADE','CNDTN','EXTWALL','ROOF','INTWALL',
                        'KITCHENS','FIREPLACES','LANDAREA',
                        'SALEDATE',
                        'NUM_UNITS','USECODE',
                        'PRICE']))

categories = ['DEISGN_STYL','ROOF','EXTWALL','INTWALL','AC','USECODE']
