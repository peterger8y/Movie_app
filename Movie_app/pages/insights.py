# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
df_app = pd.read_csv('https://raw.githubusercontent.com/peterger8y/Movie_app/main/Movie_app/assets/df_app_pdp.csv')
indicator = df_app.select_dtypes('number').columns
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
            value = 'actors',
            className='mb=5',
        ),

        dcc.Markdown('#This portion of the app allows you to see partial dependecy plots for each numeric feature:', className='mb-5'),
        
        dcc.Graph(id='indicator-graphic'),
        
        


        
    

    ]
)

layout = dbc.Row([column1])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('feature to pdp', 'value'),
     ],
    )
def update_graph(xaxis_column_name):
    dff = df_app[df_app['indicator'] == xaxis_column_name]

    title1 = 'pdp for ' + xaxis_column_name

    fig = px.line(dff, x = xaxis_column_name,
                  y = 'pred', title = title1)                     
                     
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name)

    fig.update_yaxes(title='revenue prediction')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
                     

