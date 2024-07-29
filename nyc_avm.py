import polars as pl
import xgboost as xgb
import glob
import os

resi_fn = '~/Documents/Github/dcavm/nyc_data/rollingsales_manhattan.xlsx'
resi_fns = glob.glob(os.path.expanduser('~/Documents/Github/dcavm/nyc_data/*.xlsx'))

resi_stub = os.path.expanduser('~/Documents/Github/dcavm/nyc_data/20')
first_fns = [glob.glob(resi_stub+suf) for suf in ['0*','10*']]
first_fns = [glob.glob(resi_stub+suf) for suf in ['0*','10*']]

skiprows = int('BOROUGH' in next(open('/Users/steven/Documents/Github/dcavm/nyc_data/2010_statenisland.xls')))

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

dats.columns

# TODO: make relative (e.g., 5th - 95th percentile of sales prices)
dats = dats.filter(pl.col('SALE PRICE')>1e5)
dats = dats.filter(pl.col('SALE PRICE')<2e6)

dats = dats.with_columns(
    pl.col('yearbuilt').replace(0,None),
    pl.col('yearalter1').replace(0,None),
    pl.col('yearalter2').replace(0,None)
)


xgb_data = dats.select(pl.col(['SALE PRICE',
'SALE DATE',
'yearbuilt',
'yearalter1',
'yearalter2',
'bldgclass',
'landuse',
'ltdheight',]))

categories = ['landuse','ltdheight','bldgclass']
dummies = xgb_data.select(pl.col(categories)).to_dummies(drop_first=True)

xgb_data = xgb_data.select(pl.col(set(xgb_data.columns) - set(categories)))
xgb_data = pl.concat([xgb_data,dummies],how='horizontal')
xgb_err = pl.DataFrame()
models = []

for iteration in range(1,10):
    # Date filtering for train/test

    saledate_min = pl.col('SALE DATE').min()
    saledate_max = pl.col('SALE DATE').max()

    iter_train_end_offset_str = '-%dmo' % iteration
    iter_train_end_offset = saledate_max.dt.offset_by(iter_train_end_offset_str)

    iter_test_end_offset_str = '-%dmo' % (iteration - 1)
    iter_test_end_offset = saledate_max.dt.offset_by(iter_test_end_offset_str)

    train_filter = pl.col('SALE DATE').is_between(saledate_min,
                                                 iter_train_end_offset)
    test_filter = pl.col('SALE DATE').is_between(iter_train_end_offset,
                                                iter_test_end_offset)
    
    train_data = xgb_data.filter(train_filter)
    train_data = train_data.with_columns(pl.col('SALE DATE').cast(pl.Float64))
    train_label = train_data.select('SALE PRICE')
    train_data = train_data.drop('SALE PRICE')

    test_data = xgb_data.filter(test_filter)
    test_data = test_data.with_columns(pl.col('SALE DATE').cast(pl.Float64))
    test_label = test_data.select('SALE PRICE')
    test_data = test_data.drop('SALE PRICE')

    # Debug vars

    train_date_min_debug = train_data.select('SALE DATE').min().to_numpy()[0][0]
    train_date_max_debug = train_data.select('SALE DATE').max().to_numpy()[0][0]
    test_date_min_debug = test_data.select('SALE DATE').min().to_numpy()[0][0]
    test_date_max_debug = test_data.select('SALE DATE').max().to_numpy()[0][0]

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
        nowcast_data = dats.with_columns(pl.col('SALE DATE')
            .max().alias('nowcast_date'))
        nowcast_data = nowcast_data.drop('SALE PRICE')
        dnow = xgb_data.drop('SALE PRICE')
        dnow = dnow.with_columns(pl.col('SALE DATE').cast(pl.Float64))
        dnow = xgb.DMatrix(dnow)
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
        abs(pl.col('SALE PRICE').sub(pl.col('predictions')))
        .truediv(pl.col('SALE PRICE')))
    xgb_err = pl.concat([xgb_err,test_data],how='diagonal')
