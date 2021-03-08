# Imports from 3rd party libraries
import dash
from joblib import load
pipeline = load('/Users/petergeraghty/Movie_app/assets/pipline-2.joblib')
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/peterger8y/Movie_app/main/Movie_app/assets/first_exp-5')
available_indicators = df['original_language'].value_counts().index
available_indicators2 = df['production_company'].value_counts().index
available_indicators3 = df['director'].value_counts().index
available_indicators4 = df['genres'].value_counts().index
available_indicators5 = df['production_countries'].value_counts().index
index = 0
# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown('## Predictions', className='mb-5'), 
        dcc.Markdown('#### Original_Language'), 
        dcc.Dropdown(
            id='Original_Language', 
            options = [{'label': i, 'value':i} for i in available_indicators ], 
            value = 'en', 
            className='mb-5', 
            
        ), 
        
        dcc.Markdown('#### Production_company'), 
        dcc.Dropdown(
            id='production_company', 
            options = [{'label': i, 'value':i} for i in available_indicators2],
            value = 'Universal Pictures', 
            className='mb-5', 
        ), 
        
	dcc.Markdown('#### Top Director'),
        dcc.Dropdown(
            id='top_director',
            options = [{'label': i, 'value':i} for i in available_indicators3],
            value = 1,
            className='mb-5',
	),
	
	dcc.Markdown('#### Genre'),
        dcc.Dropdown(
            id='genre',
            options = [{'label': i, 'value':i} for i in available_indicators4],
            value = 'Drama',
            className='mb-5',
        ),

	dcc.Markdown('#### Production Country'),
        dcc.Dropdown(
            id='production_country',
            options = [{'label': i, 'value':i} for i in available_indicators5],
            value = 'United States of America',
            className='mb-5',
        ),

        dcc.Markdown('#### Budget (dollars)'),
        dcc.Input(id="budget", placeholder=1000 , type="number"),
        html.Br(),
        html.P(id="output"),


        dcc.Markdown('#### Duration (minutes)'),
	dcc.Input(id="duration", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

	dcc.Markdown('#### Number of Top Actors (refer to https://www.imdb.com/list/ls058011111/'),
	dcc.Input(id="top_actors", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

        dcc.Markdown('#### Year Released'),
	dcc.Input(id="year", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

        dcc.Markdown('Month of Year released (numeric)'),
	dcc.Input(id="month", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

	dcc.Markdown('#### Day of Month Released (numeric)'),
	dcc.Input(id="day", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

        dcc.Markdown('#### Number of Languages film translated to'),
	dcc.Input(id="number_languages", placeholder=1, type="number"),
        html.Br(),
        html.P(id="output"),

	
    ],
    md=4,	
)



column2 = dbc.Col(
    [
        html.H2('Expected Gross Revenue', className='mb-5'), 
        html.Div(id='prediction-content', className='lead')
    ]
)

layout = dbc.Row([column1, column2])

import pandas as pd

@app.callback(
    Output('prediction-content', 'children'),
    [Input('Original_Language', 'value'),
     Input('production_company', 'value'),
     Input('top_director', 'value'),
     Input('genre', 'value'),
     Input('production_country', 'value'),
     Input('budget', 'value'),
     Input('duration', 'value'),
     Input('top_actors', 'value'),
     Input('year', 'value'),
     Input('month', 'value'),
     Input('day', 'value'),
     Input('number_languages', 'value'),
],
)
def predict(Original_Language, production_company, Top_director, Genre, production_country, budget,
	    duration, top_actors, year, month, day, number_languages):
    df = pd.DataFrame(
        columns=['budget', 'genres', 'original_language', 'production_countries', 'runtime',
       'director', 'production_company', 'actors', 'year', 'month', 'day',
       'number_of_languages'],
        data=[[budget, Genre, Original_Language, production_country, duration,
	       Top_director, production_company, top_actors, year, month, day, number_languages]]	  
    )
    y_pred = pipeline.predict(df)[0]
    return f'{y_pred:.0f} dollars'
