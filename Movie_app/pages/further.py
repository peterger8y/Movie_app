# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
df_app_object = pd.read_csv('https://raw.githubusercontent.com/peterger8y/Movie_app/main/Movie_app/assets/hello.csv')
indicator = ['original_language', 'production_countries', 'production_company']
import plotly.express as px


# Imports from this application
from app import app

# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown('##Feature to PDP', className = 'mb-5'),
        dcc.Dropdown(
            id = 'feature to pdp',
            options =[{'label': i, 'value': i} for i in indicator],
            value = 'production_company',
            className='mb=5',
        ),

        dcc.Markdown('#This portion of the app allows you to see box plots for pdp data for categorical features:', className='mb-5'),

        
        
        
        dcc.Graph(id='indicator-graphic1'),

        
    

    ]
)


layout = dbc.Row([column1])

@app.callback(
    Output('indicator-graphic1', 'figure'),
    [Input('feature to pdp', 'value'),
     ],
    )
def update_graph1(xaxis_column_name):
    dff1 = df_app_object[df_app_object['indicator'] == xaxis_column_name]

    title1 = 'box plot for ' + xaxis_column_name

    fig = px.box(dff1, x = xaxis_column_name,
                  y = 'pred', title = title1)                     

    fig.update_xaxes(title=xaxis_column_name)

    fig.update_layout(yaxis_range=[-10000000,200000000])

    fig.update_yaxes(title='revenue prediction')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
                     
