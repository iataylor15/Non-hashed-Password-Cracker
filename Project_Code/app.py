# Import required libraries
import os
from random import randint
import flask
import dash
from dash.dependencies import Input, Output, State
from controllers.main_app import modeler, COLS
import dash_table
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html

# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

# The main app layout
app.layout = html.Div([
    html.Center(
        children=[
            html.H1('Password Cracking Model'),
            html.Br(),
            html.A('GitHub Repo Link', href=' https://github.com/iataylor15/Non-hashed-Password-Cracker.git',
                   target='_blank')
        ]
    ),
    html.Div([
        dcc.Input(id='password', type='text', value='abc'),
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
        dcc.RadioItems(
            id='model-radio',
            options=[
                {'label': 'Insert password before crack', 'value': '1'},
                {'label': 'Try the model as is', 'value': '0'}
            ],
            value='1',
            labelStyle={'display': 'inline-block'}
        ),
        html.P(id='output-text'),
        html.Hr(),
        dash_table.DataTable(
            id='table-editing-simple',
            columns=(
                    [{'id': 'Result', 'name': 'Result'}] +
                    [{'id': p, 'name': p} for p in COLS]
            ),
            data=[
                dict(Model=i, **{param: 0 for param in COLS})
                for i in range(0)
            ],
            editable=True,
            row_deletable=True
        )
        , dcc.Loading(id="loading-icon",
                      children=[html.Div(dcc.Graph(id='main-graph'))], type="default")
    ])
])


def display_output(result):
    df = result
    fig = px.scatter(df, x='actual_tries', y='predicted_tries', color='Result')
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    return fig


@app.callback(Output('output-text', 'children'),
              [Input('model-radio', 'value')])
def update_msg(choice):
    if choice == '1':
        return f'The password is now guaranteed to be found.'
    else:
        return f'The password MAY not be found.'


@app.callback(
    Output('table-editing-simple', 'data'),
    Output('main-graph', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('table-editing-simple', 'data'),
    State('table-editing-simple', 'columns'),
    State('password', 'value'),
    State('model-radio', 'value'))
def update_output(n_clicks, rows, columns, pwd, choice):
    result = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    # remove whitespace
    if n_clicks > 0:

        pwd = "".join(pwd.split())
        if len(pwd) < 2:
            pwd = 'abc-default-pwd'
        if choice == '1':
            modeler.insert_password(pwd)
            new_data = modeler.search(pwd)
            new_data['Result'] = int(len(result) + 1)
            result = result.append(new_data, ignore_index=True)
        else:
            new_data = modeler.search(pwd)
            new_data['Result'] = int(len(result) + 1)
            result = result.append(new_data, ignore_index=True)
    return result.to_dict('records'), display_output(result)


# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
