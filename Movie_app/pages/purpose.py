# imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app

# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
	
        dcc.Markdown(
            """      
Potential Uses:

This app is intended for use by larger, experienced corporations to see what changes to a film productions features result in potential revenue outputs.
It includes a prediction function to check projected revenues with given feature inputs. Using these in conjunction with a budget can provide the means for a straightfoward 
return on investment calculation.

The Movie_app, (or any other program/app like it) should not be used for final production decisions. Three features included in the original data set that were highly predictive, but excluded
due to potential feature leakage, had to do with popularity and critical review. Therefore, any major production film, in addition to research and modeling projections, should include 
a distinctive and powerful creative and powerful creative vision. Suffice to say, films can be over-analyzed and over produced. 

Please enjoy the use of this app!

           
            """
        ),

    ],
)

layout = dbc.Row([column1])

