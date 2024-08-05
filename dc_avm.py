import polars as pl
import xgboost as xgb

def prep_dc(){
    resi_fn = '~/Documents/Github/dcavm/dc_data/Computer_Assisted_Mass_Appraisal_-_Residential.csv'
    dat = pl.read_csv(resi_fn)
    condo_fn = '~/Documents/Github/dcavm/dc_data/Computer_Assisted_Mass_Appraisal_-_Condominium.csv'
    condo = pl.read_csv(condo_fn)
    condo = condo.rename({'LIVING_GBA':'GBA'})
    gis_fn = '~/Documents/Github/dcavm/dc_data/Address_Points.csv'
    gis = pl.read_csv(gis_fn,
        columns=['LATITUDE','LONGITUDE','SSL','ADDRESS','RESIDENTIAL_TYPE'])
    address_fn = '~/Documents/Github/dcavm/dc_data/Address_Residential_Units.csv'
    address_residential_units = pl.read_csv(address_fn,
        columns=['PRIMARY_ADDRESS','CONDO_SSL','FULL_ADDRESS'])

    condo = condo.join(address_residential_units,left_on='SSL',right_on='CONDO_SSL')
    condo = condo.join(gis,left_on='PRIMARY_ADDRESS',right_on='ADDRESS')

    condo = condo.filter(pl.col('RESIDENTIAL_TYPE')=='RESIDENTIAL')
    dat = dat.filter(pl.col('QUALIFIED')=='Q')
    condo = condo.filter(pl.col('QUALIFIED')=='Q')

    dat = dat.join(gis,on='SSL')

    condo = condo.rename({'FULL_ADDRESS':'ADDRESS'})
    condo = condo.select(pl.col(set(condo.columns) & set(dat.columns)))
    dat = pl.concat([dat,condo], how='diagonal_relaxed')

    dat = dat.filter(pl.col('PRICE')>1e5)
    dat = dat.filter(pl.col('PRICE')<2e6)
    dat = dat.with_columns(
        pl.col('EYB').replace(0,None),
        pl.col('AYB').replace(0,None)
    )

    dat = dat.with_columns(
    pl.col("SALEDATE").str.to_date("%Y/%m/%d %H:%M:%S+00")
    )

    dat = dat.with_columns(
    pl.col("AC")=='Y'
    )

    other_col_name_array = ['LATITUDE','LONGITUDE',
                'BATHRM','HF_BATHRM','HEAT','AC','ROOMS',
                                'BEDRM','AYB','YR_RMDL','EYB','STORIES','GBA',
                                'GRADE','CNDTN','EXTWALL','ROOF','INTWALL',
                                'KITCHENS','FIREPLACES','LANDAREA',
                                'NUM_UNITS','USECODE',]
    categorical_name_array = ['HEAT','ROOF','EXTWALL','INTWALL','AC','USECODE']
    price_col_name = 'PRICE'
    date_col_name = 'SALEDATE'
}

xgb_data = data_prep_for_model(dat,date_col_name,price_col_name,other_col_name_array,categorical_name_array)
model_return = fit_model(xgb_data,date_col_name,price_col_name)