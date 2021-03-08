# Imports from 3rd party libraries
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
            Eval metric choice:
            
            For the stated purpose of this model, I selected R2 as my evaluation metric. This is because the problem, as stated in the purpose section,
            is a regression task. Maximizing R2 minimizes variance in predictions.
            
            Data Cleaning and feature production:

            First, some for loops were written to clean up some of the messy and uninterpretable object features.
            Features like 'actors' and 'genre' were nearly unreadable, and would have made a headache for future interpretation.
            I referenced another similar dataset to increase the number of potential features for my algorithms. I was able to include a list of
            the main cast, along with directors in my final data set from this alternate dataset.
            
            Second, some features were referenced to an outside dataframe to make them easier to interpet to my employed learning algorithms.
            Benefits to my R2 were found in referencing a top actors and directors lists for by actors and directors list, respectively.
            Instead of using specific actor names, I translated to the number of top actors, and a binary of whether or not a film was run by a
            top director, according to their respective IMDB dataframes.

            In addition to this, I cleaned and removed values of 0 for my budget column. My R2 actually decreased,
            but I feel this change is essential for model interpretability to a potential application user.
            Rows with a revenue of zero were simply dropped from my dataframe; the purpose of this app is to
            predict revenue outputs. With no revenue, an observation is worthless for training purposes.
            

            Data Simplification:

            Simplification was more important for my Linear model than my decision tree, but both were helped with simplification.
            Instead of using a list of languages a film was released in, I simply counted the number of languages listed. this had
            the greatest effect on my linear model.

            Hyperparamater Tuning:

            My linear regression performed horribly at first: The R2 was negative.... many features had to be dropped due to high cardinality.
            However, with a little more cleaning and a substitution of a ridge regression, my linear model performance jumped up. In terms of R2,
            the linear model wasnt close to the Random forest regressor. I've neglected to include the coefficients for this model, as the range of featuers was
            quite sparse, and the r2 was quite low.

            My main model, the decision tree regressor, had a performance improvement from increasing the number of
            trees as well as increasing the minimum sample size of each leaf. 
            


            """
        ),

        

    ],
)

layout = dbc.Row([column1])
