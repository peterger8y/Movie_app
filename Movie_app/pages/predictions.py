# Imports from 3rd party libraries
import dash
from joblib import load
pipeline = load('https://github.com/peterger8y/Movie_app/blob/3dc141e7997a11ec4378af78d1f1d19061aa2b67/assets/pipline-2.joblib')
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/peterger8y/Movie_app/master/first_exp')
available_indicators = df['original_language'].value_counts().index
available_indicators2 = df['production_company'].value_counts().index
available_indicators3 = df['director'].value_counts().index
available_indicators4 = df['genres'].value_counts().index
available_indicators5 = df['production_countries'].value_counts().index
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
            value = 'no',
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


        dbc.Input(id="budget", placeholder="Input Film Budget", type="number"),
        html.Br(),
        html.P(id="output"),
	
	dbc.Input(id="vote_average", placeholder="Input film vote_average, imdb system, 1 -10", type="number"),
        html.Br(),
        html.P(id="output"),

	dbc.Input(id="vote_count", placeholder="Input film vote_count, total votes cast", type="number"),
        html.Br(),
        html.P(id="output"),

	dbc.Input(id="duration", placeholder="Input film duration(minutes)", type="number"),
        html.Br(),
        html.P(id="output"),
	
	dbc.Input(id="top_actors", placeholder="input number of top actors", type="number"),
        html.Br(),
        html.P(id="output"),

	dbc.Input(id="year", placeholder="Input year released", type="number"),
        html.Br(),
        html.P(id="output"),

	dbc.Input(id="month", placeholder="Input month of year(number)", type="number"),
        html.Br(),
        html.P(id="output"),
	
	dbc.Input(id="day", placeholder="Input day of month", type="number"),
        html.Br(),
        html.P(id="output"),

	dbc.Input(id="number_languages", placeholder="Input number of languages released", type="number"),
        html.Br(),
        html.P(id="output"),
	
	dbc.Input(id="popularity", placeholder="popularity rating", type="number"),
        html.Br(),
        html.P(id="output"),
	
    ],
    md=4,	
)



column2 = dbc.Col(
    [
        html.H2('Expected Lifespan', className='mb-5'), 
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
     Input('budget', 'output'),
     Input('vote_average', 'output'),
     Input('vote_count', 'output'),
     Input('duration', 'output'),
     Input('top_actors', 'output'),
     Input('year', 'output'),
     Input('month', 'output'),
     Input('day', 'output'),
     Input('number_languages', 'output'),
     
],
)
def predict(Original_Language, production_company, Top_director, Genre, production_country, budget,
	    vote_average, vote_count, duration, top_actors, year, month, day, number_languages, popularity):
    df = pd.DataFrame(
        columns=['budget', 'genres' 'original_language', 'popularity', 'production_countries', 'runtime',
		 'vote_average', 'vote_count', 'duration', 'director', 'production_company', 'actors', 'year', 
	         'month', 'day', 'number_of_languages'],
        data=[[budget, Genre, Original_Language, popularity, production_company, duration, vote_average, vote_count,
	       duration, Top_director, production_company, top_actors, year, month, day, number_languages]]	  
    )
    y_pred = pipeline.predict(df)[0]
    return f'{y_pred:.0f} years'
