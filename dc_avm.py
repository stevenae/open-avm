import polars as pl
import xgboost as xgb

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
xgb_data = data_prep_for_model(dat,date_col_name,price_col_name,other_col_name_array,categorical_name_array)
model_return = fit_model(xgb_data,date_col_name,price_col_name)

region_name = 'dc'
if(0):
    xgb_data = dat.select(pl.col(['LATITUDE','LONGITUDE',
            'BATHRM','HF_BATHRM','HEAT','AC','ROOMS',
                            'BEDRM','AYB','YR_RMDL','EYB','STORIES','GBA',
                            'GRADE','CNDTN','EXTWALL','ROOF','INTWALL',
                            'KITCHENS','FIREPLACES','LANDAREA',
                            'SALEDATE',
                            'NUM_UNITS','USECODE',
                            'PRICE']))

    categories = ['HEAT','ROOF','EXTWALL','INTWALL','AC','USECODE']
    dummies = xgb_data.select(pl.col(categories)).to_dummies(drop_first=True)
    xgb_data = xgb_data.select(pl.col(set(xgb_data.columns) - set(categories)))
    xgb_data = pl.concat([xgb_data,dummies],how='horizontal')
    xgb_err = pl.DataFrame()
    models = []
    for iteration in range(1,10):
        # Date filtering for train/test

        saledate_min = pl.col('SALEDATE').min()
        saledate_max = pl.col('SALEDATE').max()

        iter_train_end_offset_str = '-%dmo' % iteration
        iter_train_end_offset = saledate_max.dt.offset_by(iter_train_end_offset_str)

        iter_test_end_offset_str = '-%dmo' % (iteration - 1)
        iter_test_end_offset = saledate_max.dt.offset_by(iter_test_end_offset_str)

        train_filter = pl.col('SALEDATE').is_between(saledate_min,
                                                    iter_train_end_offset)
        test_filter = pl.col('SALEDATE').is_between(iter_train_end_offset,
                                                    iter_test_end_offset)
        
        train_data = xgb_data.filter(train_filter)
        train_label = train_data.select('PRICE')
        train_data = train_data.drop('PRICE')

        test_data = xgb_data.filter(test_filter)
        test_label = test_data.select('PRICE')
        test_data = test_data.drop('PRICE')

        # Debug vars

        train_date_min_debug = train_data.select('SALEDATE').min().to_numpy()[0][0]
        train_date_max_debug = train_data.select('SALEDATE').max().to_numpy()[0][0]
        test_date_min_debug = test_data.select('SALEDATE').min().to_numpy()[0][0]
        test_date_max_debug = test_data.select('SALEDATE').max().to_numpy()[0][0]

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

        if iteration == 1:
            # nowcast predictions and comps
            nowcast_data = dat.with_columns(pl.col('SALEDATE')
                .max().alias('nowcast_date'))
            nowcast_data = nowcast_data.drop('PRICE')
            dnow = xgb.DMatrix(xgb_data.drop('PRICE'))
            nowcast_predictions = bst.predict(dnow)
            nowcast_data = nowcast_data.with_columns(
                nowcast_prediction = nowcast_predictions
            )
            nowcast_data.write_csv('~/Documents/nowcast_predictions.csv',
                separator=",")

        models.append(bst)

        predictions = pl.DataFrame({'predictions' : bst.predict(dtest)})
        test_data = pl.concat([test_data,predictions,test_label],how='horizontal')
        test_data = test_data.with_columns(error =
            abs(pl.col('PRICE').sub(pl.col('predictions')))
            .truediv(pl.col('PRICE')))
        xgb_err = pl.concat([xgb_err,test_data],how='diagonal')

    xgb_err.write_csv('~/Documents/xgb_errors.csv', separator=",")
