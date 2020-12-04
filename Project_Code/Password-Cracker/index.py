import dash_html_components as html

from appview import password_cracker
from controllers.main_app import app, modeler


# Show Password Cracker View
def display_page():
    return password_cracker.layout


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
    html.Div(id='page-content', children=display_page())
])

if __name__ == '__main__':
    modeler.foo
    app.run_server(debug=False)
