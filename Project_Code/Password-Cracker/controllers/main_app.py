import dash

from models.ModelBuilder import ModelBuilder

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


modeler = ModelBuilder()
COLS = ['pwd', 'found', 'predicted_tries', 'actual_tries']
selection = 1
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
