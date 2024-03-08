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
        preds_row = preds.filter(pl.col('ADDRESS')==input.selectize())
        preds_row.describe()
        return render.DataGrid(preds_row, row_selection_mode="single")


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