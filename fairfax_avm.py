import polars as pl
import xgboost as xgb

resi_fn = '~/Documents/Github/dcavm/fairfax_data/Tax_Administration_s_Real_Estate_-_Sales_Data.csv'
dat = pl.read_csv(resi_fn,
                  columns=['PARID','SALEDT','PRICE','SALEVAL_DESC'])
# dat['SALEVAL_DESC'].value_counts().sort(by='count').tail(10)
dat = dat.filter(pl.col('SALEVAL_DESC')=='Valid and verified sale')

gis_fn = '~/Documents/Github/dcavm/fairfax_data/Address_Points.csv'
gis = pl.read_csv(gis_fn,
    columns=['x','y','Parcel Identification Number'])

dwelling_fn = '~/Documents/Github/dcavm/fairfax_data/Tax_Administration_s_Real_Estate_-_Dwelling_Data.csv'
dwelling = pl.read_csv(dwelling_fn,
    columns=['PARID','YRBLT','EFFYR','YRREMOD','RMBED','FIXBATH','FIXHALF','RECROMAREA','WBFP_PF','BSMTCAR','GRADE_DESC','SFLA','BSMT_DESC','CDU_DESC','EXTWALL_DESC','HEAT_DESC','USER13_DESC'])
dwelling = dwelling.rename({'USER13_DESC':'ROOF'})

dat = dat.join(gis,left_on='PARID',right_on='Parcel Identification Number')
dat = dat.join(dwelling,left_on='PARID',right_on='PARID')

dat = dat.filter(pl.col('PRICE')>1e5)
dat = dat.filter(pl.col('PRICE')<2e6)

dat = dat.with_columns(
   pl.col("SALEDT").str.to_date("%Y/%m/%d %H:%M:%S+00")
)

xgb_data = dat.select(pl.col(['x','y',
            'YRBLT','EFFYR','YRREMOD','RMBED','FIXBATH','FIXHALF','RECROMAREA','WBFP_PF','BSMTCAR','GRADE_DESC','SFLA','BSMT_DESC','CDU_DESC','EXTWALL_DESC','HEAT_DESC','ROOF',
                        'SALEDT',
                        'PRICE']))

categories = ['GRADE_DESC','ROOF','BSMT_DESC','CDU_DESC','EXTWALL_DESC','HEAT_DESC']
dummies = xgb_data.select(pl.col(categories)).to_dummies(drop_first=True)
xgb_data = xgb_data.select(pl.col(set(xgb_data.columns) - set(categories)))
xgb_data = pl.concat([xgb_data,dummies],how='horizontal')
xgb_err = pl.DataFrame()
models = []
for iteration in range(1,10):
    # Date filtering for train/test

    saledate_min = pl.col('SALEDT').min()
    saledate_max = pl.col('SALEDT').max()

    iter_train_end_offset_str = '-%dmo' % iteration
    iter_train_end_offset = saledate_max.dt.offset_by(iter_train_end_offset_str)

    iter_test_end_offset_str = '-%dmo' % (iteration - 1)
    iter_test_end_offset = saledate_max.dt.offset_by(iter_test_end_offset_str)

    train_filter = pl.col('SALEDT').is_between(saledate_min,
                                                 iter_train_end_offset)
    test_filter = pl.col('SALEDT').is_between(iter_train_end_offset,
                                                iter_test_end_offset)
    
    train_data = xgb_data.filter(train_filter)
    train_label = train_data.select('PRICE')
    train_data = train_data.drop('PRICE')

    test_data = xgb_data.filter(test_filter)
    test_label = test_data.select('PRICE')
    test_data = test_data.drop('PRICE')

    # Debug vars

    train_date_min_debug = train_data.select('SALEDT').min().to_numpy()[0][0]
    train_date_max_debug = train_data.select('SALEDT').max().to_numpy()[0][0]
    test_date_min_debug = test_data.select('SALEDT').min().to_numpy()[0][0]
    test_date_max_debug = test_data.select('SALEDT').max().to_numpy()[0][0]

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
        nowcast_data = dat.with_columns(pl.col('SALEDT')
            .max().alias('nowcast_date'))
        nowcast_data = nowcast_data.drop('PRICE')
        dnow = xgb.DMatrix(xgb_data.drop('PRICE'))
        nowcast_predictions = bst.predict(dnow)
        nowcast_data = nowcast_data.with_columns(
            nowcast_prediction = nowcast_predictions
        )
        nowcast_data.write_csv('~/Documents/fairfax_nowcast_predictions.csv',
            separator=",")

    models.append(bst)

    predictions = pl.DataFrame({'predictions' : bst.predict(dtest)})
    test_data = pl.concat([test_data,predictions,test_label],how='horizontal')
    test_data = test_data.with_columns(error =
        abs(pl.col('PRICE').sub(pl.col('predictions')))
        .truediv(pl.col('PRICE')))
    xgb_err = pl.concat([xgb_err,test_data],how='diagonal')

xgb_err.write_csv('~/Documents/fairfax_xgb_errors.csv', separator=",")
