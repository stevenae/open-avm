import polars as pl
import xgboost as xgb

def price_filter(dat,price_col_name,low_filter,high_filter):
    dat = dat.filter(pl.col(price_col_name)>low_filter)
    dat = dat.filter(pl.col(price_col_name)<high_filter)
    return dat

def date_format(dat,date_col_name,fmt_str):
    dat = dat.with_columns(
        pl.col(date_col_name).str.to_date(fmt_str)
    )
    return dat

def data_prep_for_model(dat,date_col_name,price_col_name,numerical_col_name_array,categorical_name_array):
    all_cols = [price_col_name,date_col_name,*numerical_col_name_array,*categorical_name_array]
    xgb_data = dat.select(pl.col(all_cols))
    xgb_data = xgb_data.with_columns(pl.col(numerical_col_name_array).cast(pl.Float64))
    xgb_data = xgb_data.with_columns(pl.col(price_col_name).cast(pl.Float64))
    dummies = xgb_data.select(pl.col(categorical_name_array)).to_dummies(drop_first=True)
    xgb_data = xgb_data.select(pl.col(set(xgb_data.columns) - set(categorical_name_array)))
    xgb_data = pl.concat([xgb_data,dummies],how='horizontal')

    return [xgb_data,date_col_name,price_col_name]

def fit_model(xgb_data,date_col_name,price_col_name):
    models = []
    xgb_err = pl.DataFrame()
    print(xgb_data[date_col_name])
    for iteration in range(1,10):
        # Date filtering for train/test

        saledate_min = pl.col(date_col_name).min()
        saledate_max = pl.col(date_col_name).max()

        iter_train_end_offset_str = '-%dmo' % iteration
        iter_train_end_offset = saledate_max.dt.offset_by(iter_train_end_offset_str)

        iter_test_end_offset_str = '-%dmo' % (iteration - 1)
        iter_test_end_offset = saledate_max.dt.offset_by(iter_test_end_offset_str)

        train_filter = pl.col(date_col_name).is_between(saledate_min,
                                                    iter_train_end_offset)
        test_filter = pl.col(date_col_name).is_between(iter_train_end_offset,
                                                    iter_test_end_offset)
        
        train_data = xgb_data.filter(train_filter)
        train_data = train_data.with_columns(pl.col(date_col_name).cast(pl.Float64))
        train_label = train_data.select(price_col_name)
        train_data = train_data.drop(price_col_name)

        test_data = xgb_data.filter(test_filter)
        test_data = test_data.with_columns(pl.col(date_col_name).cast(pl.Float64))
        test_label = test_data.select(price_col_name)
        test_data = test_data.drop(price_col_name)

        # Debug vars

        train_date_min_debug = train_data.select(date_col_name).min().to_numpy()[0][0]
        train_date_max_debug = train_data.select(date_col_name).max().to_numpy()[0][0]
        test_date_min_debug = test_data.select(date_col_name).min().to_numpy()[0][0]
        test_date_max_debug = test_data.select(date_col_name).max().to_numpy()[0][0]

        print('iter {}'.format(iteration))
        print('train {} to {}'.format(train_date_min_debug,train_date_max_debug))
        print('test {} to {}'.format(test_date_min_debug,test_date_max_debug))
        print('%d tr obs, %d va obs' % (len(train_label), len(test_label)))

        dtrain = xgb.DMatrix(train_data, label = train_label)
        dtest = xgb.DMatrix(test_data, label = test_label)
        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        param = {'max_depth': 10, 'eta': .01, 'objective': 'reg:tweedie',
            'eval_metric':'mape', 'tree_method':'hist', 'grow_policy':'lossguide'}
        num_round = 10000
        bst = xgb.train(param, dtrain, num_round, evals=evallist,
            early_stopping_rounds=100, verbose_eval=100)

        models.append(bst)

        predictions = pl.DataFrame({'predictions' : bst.predict(dtest)})
        test_data = pl.concat([test_data,predictions,test_label],how='horizontal')
        test_data = test_data.with_columns(error =
            abs(pl.col(price_col_name).sub(pl.col('predictions')))
            .truediv(pl.col(price_col_name)))
        xgb_err = pl.concat([xgb_err,test_data],how='diagonal')

    return_dict = {'models':models,
                    'test_data':test_data,
                    'xgb_err':xgb_err}
    return return_dict

# TODO
def generate_nowcast(bst,dats,xgb_data,region_name,date_col_name,price_col_name):
    # nowcast predictions and comps
    nowcast_data = dats.with_columns(pl.col(date_col_name)
        .max().alias('nowcast_date'))
    nowcast_data = nowcast_data.drop(price_col_name)
    dnow = xgb_data.drop(price_col_name)
    dnow = dnow.with_columns(pl.col(date_col_name).cast(pl.Float64))
    dnow = xgb.DMatrix(dnow)
    nowcast_predictions = bst.predict(dnow)
    nowcast_data = nowcast_data.with_columns(
        nowcast_prediction = nowcast_predictions
    )
    nowcast_data.write_csv('~/Documents/Github/dcvam/'+region_name+'_nowcast_predictions.csv',
        separator=",")

def prep_dc(*args):
    resi_fn = '~/Documents/Github/dcavm/dc_data/Computer_Assisted_Mass_Appraisal_-_Residential.csv'
    dat = pl.read_csv(resi_fn)
    condo_fn = '~/Documents/Github/dcavm/dc_data/Computer_Assisted_Mass_Appraisal_-_Condominium.csv'
    condo = pl.read_csv(condo_fn)
    condo = condo.rename({'LIVING_GBA': 'GBA'})
    gis_fn = '~/Documents/Github/dcavm/dc_data/Address_Points.csv'
    gis = pl.read_csv(gis_fn,
                      columns=['LATITUDE', 'LONGITUDE', 'SSL', 'ADDRESS', 'RESIDENTIAL_TYPE'])
    address_fn = '~/Documents/Github/dcavm/dc_data/Address_Residential_Units.csv'
    address_residential_units = pl.read_csv(address_fn,
                                            columns=['PRIMARY_ADDRESS', 'CONDO_SSL', 'FULL_ADDRESS'])

    condo = condo.join(address_residential_units, left_on='SSL', right_on='CONDO_SSL')
    condo = condo.join(gis, left_on='PRIMARY_ADDRESS', right_on='ADDRESS')

    condo = condo.filter(pl.col('RESIDENTIAL_TYPE') == 'RESIDENTIAL')
    dat = dat.filter(pl.col('QUALIFIED') == 'Q')
    condo = condo.filter(pl.col('QUALIFIED') == 'Q')

    dat = dat.join(gis, on='SSL')

    condo = condo.rename({'FULL_ADDRESS': 'ADDRESS'})
    condo = condo.select(pl.col(set(condo.columns) & set(dat.columns)))
    dat = pl.concat([dat, condo], how='diagonal_relaxed')

    # dat = dat.filter(pl.col('PRICE') > 1e5)
    # dat = dat.filter(pl.col('PRICE') < 2e6)

    price_col_name = 'PRICE'
    dat = price_filter(dat,price_col_name,1e5,2e6)
    
    dat = dat.with_columns(
        pl.col('EYB').replace(0, None),
        pl.col('AYB').replace(0, None)
    )

    # dat = dat.with_columns(
    #     pl.col("SALEDATE").str.to_date("%Y/%m/%d %H:%M:%S+00")
    # )
    date_col_name = 'SALEDATE'
    dat = date_format(dat,date_col_name,"%Y/%m/%d %H:%M:%S+00")

    dat = dat.with_columns(
        pl.col("AC") == 'Y'
    )

    numerical_col_name_array = ['LATITUDE', 'LONGITUDE',
                            'BATHRM', 'HF_BATHRM',  'ROOMS',
                            'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'GBA',
                            'GRADE', 'CNDTN',
                            'KITCHENS', 'FIREPLACES', 'LANDAREA',
                            'NUM_UNITS',]
    categorical_name_array = ['HEAT', 'ROOF', 'EXTWALL', 'INTWALL', 'AC', 'USECODE']

    return [dat,date_col_name,price_col_name,numerical_col_name_array,categorical_name_array]

def prep_nyc():
    resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

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

    dats = dats.join(gis,on='bbl')

    dats.columns

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
    fmt_str = "%Y-%m-%d %H:%M:%S"
    dats = date_format(dats,date_col_name,fmt_str)
    dats = dats

    return [dat, numerical_col_name_array, categorical_name_array, price_col_name, date_col_name]

