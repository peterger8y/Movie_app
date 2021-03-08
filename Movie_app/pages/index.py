# Imports from 3rd party libraries
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import seaborn as sns

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
            ## Movies - getting a bang for your buck

            Movie_App provides potential revenue of bigger budget films, estimate the effects of any particular 
            
            feature on revenues. Adjust some features of an upcoming film production to see potential impacts on revenue.      
          
           
            """
        ),
        dcc.Link(dbc.Button('Predict Your Movie Revenue Outcome!', color='primary'), href='/predictions')
    ],
    md=4,
)

df = pd.read_csv('https://raw.githubusercontent.com/peterger8y/Movie_app/main/Movie_app/assets/first_exp-5')
fig = px.scatter_matrix(df, dimensions=['runtime', 'revenue', 'number_of_languages', 'year'])

column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

layout = dbc.Row([column1, column2])
