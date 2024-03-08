from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly
import plotly.express as px
import pandas as pd
import polars as pl
from itertools import islice

preds = pl.read_csv('~/Documents/nowcast_predictions.csv')
addresses = preds['ADDRESS']
n = 10
# list(islice(filter(lambda x: theSubString in x, addresses), n))

ui.page_opts(title="Penguins dashboard", fillable=True)

with ui.sidebar():
    ui.input_selectize(  
        "selectize",  
        "Type or select an address:",  
        list(addresses)[1:10000],
    )
    @render.data_frame
    def summary_data():
        selected_row = preds.filter(pl.col('ADDRESS')==input.selectize())
        rank_cols = ['BATHRM','BEDRM','GBA','YR_RMDL']
        price_offset = 1e4
        selected_nowcast_prediction = selected_row['nowcast_prediction']
        nowcast_prediction_offset_filter = pl.col('nowcast_prediction').is_between(selected_nowcast_prediction+price_offset,
                                                                selected_nowcast_prediction-price_offset)
        similarly_priced = preds.filter(nowcast_prediction_offset_filter)
        similarly_priced = similarly_priced.select(rank_cols)
        preds_ranks = similarly_priced.select((pl.col(rank_cols).rank() / pl.col(rank_cols).count()))
        return render.DataGrid(selected_row, row_selection_mode="single",summary=False)

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