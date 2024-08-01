def rename_cols(dat,name_dict):
    dat.rename(name_dict)

    return dat

def gis_join(dat,gis,join_key):
    dat.join(gis,on=join_key)

    return dat

def price_filter(dat,price_col_name,low_filter,high_filter):
    dat = dat.filter(pl.col(price_col_name)>low_filter)
    dat = dat.filter(pl.col(price_col_name)<high_filter)

    return dat


def date_format(dat,fmt_str):
    dat = dat.with_columns(
        pl.col(date_col_name).str.to_date(fmt_str)
    )

    return dat

def data_prep_for_model(dat,date_col_name,price_col_name,other_col_name_array,categorical_name_array):
    all_cols = [price_col_name,date_col_name,*other_col_name_array]
    xgb_data = dat.select(pl.col(all_cols))

    dummies = xgb_data.select(pl.col(categorical_name_array)).to_dummies(drop_first=True)
    xgb_data = xgb_data.select(pl.col(set(xgb_data.columns) - set(categorical_name_array)))
    xgb_data = pl.concat([xgb_data,dummies],how='horizontal')

    return xgb_data

def fit_model(xgb_data,date_col_name,price_col_name,region_name):
    models = []
    xgb_err = pl.DataFrame()

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

        if iteration == 1:
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
