from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly
import plotly.express as px
import pandas as pd
import polars as pl
from itertools import islice

# Read nowcast for visualization
rank_cols = ['BATHRM','HF_BATHRM','BEDRM','GBA','YR_RMDL','EYB']
nowcast_price_col = 'nowcast_prediction'
address_col = 'ADDRESS'
nowcast_select_cols = [*rank_cols,nowcast_price_col,address_col]
preds = pl.read_csv('~/Documents/nowcast_predictions.csv',
                    columns=nowcast_select_cols)

preds = preds.with_columns(pl.col('BATHRM').add(pl.col('HF_BATHRM')))
preds = preds.drop('HF_BATHRM')
rank_cols.remove('HF_BATHRM')
addresses = preds['ADDRESS']

# Read backtest for error and comps
prev_sale_price_col = 'PRICE'
prev_sale_select_cols = [prev_sale_price_col,'SALEDATE','error','predictions']
prev_sale_select_cols = prev_sale_select_cols.extend(address_col)
errs = pl.read_csv('~/Documents/xgb_errors.csv',
                    columns=prev_sale_select_cols)

ui.page_opts(title="Penguins dashboard", fillable=True)

with ui.sidebar():
    ui.input_selectize(  
        "selectize",  
        "Type or select an address:",  
        list(addresses.drop_nans().sample(10000).sort()),
        multiple=True
    )
    @render.data_frame
    def summary_data():
        selected_rows = preds.filter(pl.col('ADDRESS').is_in(input.selectize()))
        price_offset_pct = .10
        prediction_range_filter_bottom = selected_rows['nowcast_prediction'].min()*(1-price_offset_pct)
        prediction_range_filter_top = selected_rows['nowcast_prediction'].max()*(1+price_offset_pct)
        nowcast_prediction_offset_filter = pl.col('nowcast_prediction').is_between(prediction_range_filter_bottom,
                                                                prediction_range_filter_top)
        similarly_priced = preds.filter(nowcast_prediction_offset_filter)
        # Add seleted rows to nearby homes to calculate ranks
        similarly_priced = pl.concat([similarly_priced,selected_rows],how='vertical')
        num_selected_rows = selected_rows.shape[0]
        print(similarly_priced.tail(num_selected_rows))
        similarly_priced = similarly_priced.with_columns((pl.col(rank_cols).rank() / pl.col(rank_cols).count()))
        print(similarly_priced.tail(num_selected_rows))
        return render.DataGrid(selected_rows,
                               row_selection_mode="single",
                               summary=False)

@render.text
def value():
    return f"{input.selectize()}"



@reactive.calc
def df():
    from palmerpenguins import load_penguins
    return load_penguins()[input.var()]

with ui.card(full_screen=True):
    @render_plotly
    def radar():
        
        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2, 3],
            theta=['processing cost','mechanical properties','chemical stability',
                   'thermal stability', 'device integration']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        
        return fig



# fig.show()