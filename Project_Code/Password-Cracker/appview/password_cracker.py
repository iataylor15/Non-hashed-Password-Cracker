import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State

from controllers.main_app import modeler, COLS, app

layout = html.Div([
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
