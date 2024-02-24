import polars as pl
import xgboost as xgb

dat = pl.read_csv('~/Downloads/Computer_Assisted_Mass_Appraisal_-_Residential.csv')
condo = pl.read_csv('~/Downloads/Computer_Assisted_Mass_Appraisal_-_Condominium.csv')
condo.rename({'LIVING_GBA':'GBA'})
gis = pl.read_csv('~/Downloads/Address_Points.csv',  columns=['LATITUDE','LONGITUDE','SSL','ADDRESS','RESIDENTIAL_TYPE'])
address_residential_units = pl.read_csv('~/Downloads/Address_Residential_Units.csv', columns=['PRIMARY_ADDRESS','CONDO_SSL','FULL_ADDRESS'])

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
# dat = dat.filter(pl.col('GBA')>0)
dat = dat.with_columns(
    # pl.col('GBA').replace(0,None),
    pl.col('EYB').replace(0,None),
    pl.col('AYB').replace(0,None)
)
# dat = dat.with_columns(
#     pl.col('PRICE').truediv(pl.col('GBA'))
# )

dat = dat.with_columns(
   pl.col("SALEDATE").str.to_date("%Y/%m/%d %H:%M:%S+00")
)

dat = dat.with_columns(
   pl.col("AC")=='Y'
)

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
    print('{} months back from {}'.format(iteration,xgb_data.select('SALEDATE').max().to_numpy()[0][0]))
    offset_str = '-%dmo' % iteration
    train_data = xgb_data.filter(pl.col('SALEDATE')<pl.col('SALEDATE').max().dt.offset_by(offset_str)).drop('PRICE')
    train_label = xgb_data.filter(pl.col('SALEDATE')<pl.col('SALEDATE').max().dt.offset_by(offset_str)).select('PRICE')
    test_data = xgb_data.filter(pl.col('SALEDATE')>=pl.col('SALEDATE').max().dt.offset_by(offset_str)).drop('PRICE')
    test_label = xgb_data.filter(pl.col('SALEDATE')>=pl.col('SALEDATE').max().dt.offset_by(offset_str)).select('PRICE')

    print('%d tr, %d va' % (len(train_label), len(test_label)))

    dtrain = xgb.DMatrix(train_data, label = train_label)
    dtest = xgb.DMatrix(test_data, label = test_label)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    param = {'max_depth': 10, 'eta': .01, 'objective': 'reg:tweedie', 'eval_metric':'mape', 'tree_method':'hist', 'grow_policy':'lossguide'}
    num_round = 10000
    bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=100, verbose_eval=100)

    if iteration == 1:
        # nowcast predictions and comps
        nowcast_data = dat.with_columns(pl.col('SALEDATE').max().alias('nowcast_date'))
        nowcast_data = nowcast_data.drop('PRICE')
        dnow = xgb.DMatrix(xgb_data.drop('PRICE'))
        nowcast_predictions = bst.predict(dnow)
        nowcast_data = nowcast_data.with_columns(
            # nowcast_prediction = pl.col('GBA').mul(nowcast_predictions)
            nowcast_prediction = nowcast_predictions
        )
        nowcast_data.write_csv('~/Documents/nowcast_predictions.csv', separator=",")

    models.append(bst)

    predictions = pl.DataFrame({'predictions' : bst.predict(dtest)})
    test_data = pl.concat([test_data,predictions,test_label],how='horizontal')
    test_data = test_data.with_columns(error = abs(pl.col('PRICE').sub(pl.col('predictions'))).truediv(pl.col('PRICE')))
    xgb_err = pl.concat([xgb_err,test_data],how='diagonal')

    xgb_data = xgb_data.filter(pl.col('SALEDATE')<pl.col('SALEDATE').max().dt.offset_by(offset_str))

xgb_err.write_csv('~/Documents/xgb_errors.csv', separator=",")
