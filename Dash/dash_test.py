import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import dash_table
import datetime as dt
import base64
import numpy as np
import pickle
from dash.dependencies import Input, Output, State
from model import *
import string

def generate_table(dataframe,page_size=10):
    return dash_table.DataTable(
        id='dataTable',
        columns=[{
            "name": i,
            "id":i
        }for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action="native",
        page_current=0,
        page_size=page_size,
        style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        'minWidth': '0px',
        'maxWidth': '750px'})

df = pd.read_csv('train.csv')
# df_eda = pd.read_csv('df_eda.csv')

external_stylesheets=['https://codepen.io/chriddyp/pen/bWlwgP.css']

app=dash.Dash(__name__, external_stylesheets=external_stylesheets) 

app.layout =html.Div(children=[
    html.H1('Hate Comment Detector'),
    
    html.Div(children='''
        About Hate Comment :
    '''),
    html.P(children='''
        Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions.
    '''),

    dcc.Tabs(children=[
        dcc.Tab(value='tab-1', label='Hate speech database', children=[
            html.Div(children=[
                html.Div([
                    html.P('Toxic'),
                    dcc.Dropdown(value='',
                    id='filter-toxic',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4'),
                html.Div([
                    html.P('Severe Toxic'),
                    dcc.Dropdown(value='',
                    id='filter-severe_toxic',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4'),
                html.Div([
                    html.P('Obscene'),
                    dcc.Dropdown(value='',
                    id='filter-obscene',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4')],className='row'),
            html.Br(),
            html.Div(children=[
                html.Div([
                    html.P('Threat'),
                    dcc.Dropdown(value='',
                    id='filter-threat',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4'),
                html.Div([
                    html.P('Insult'),
                    dcc.Dropdown(value='',
                    id='filter-insult',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4'),
                html.Div([
                    html.P('Identity Hate'),
                    dcc.Dropdown(value='',
                    id='filter-ihate',
                    options=[
                    {'label':'No','value':0},
                    {'label':'Yes','value':1},
                    {'label':'None','value':''}])],className='col-4')],className='row'),
            html.Br(),
            html.Div([
                html.Div([
                    html.P('Max Rows'),
                    dcc.Input(id='filter-row',
                            type='number',
                            value=10)],className='col-6'),
                html.Div([
                    html.P('Search Keywords'),
                    dcc.Input(id='filter-keyword',
                            type='text',
                            value='')],className='col-6')],className='row'),
            html.Br(),
            html.Br(),
            html.Div([html.Button('Search', id='filter-df')]),
            html.Br(),
            html.Div(id='div-table', children=[
                generate_table(df,page_size=10)],className='col-10')]),
        dcc.Tab(value='tab-2', label='EDA', children=[
            html.H4('Histogram for information inside text'),
            html.Div(children=[
                html.Div([
                    html.P('Hate type'),
                    dcc.Dropdown(value='toxic',
                    id='filter-hate-type',
                    options=[
                    {'label':'Toxic','value':'toxic'},
                    {'label':'Severe Toxic','value':'severe_toxic'},
                    {'label':'Obscene','value':'obscene'},
                    {'label':'Threat','value':'threat'},
                    {'label':'Insult','value':'insult'},
                    {'label':'Identity Hate','value':'identity_hate'},
                    {'label':'Not Hate','value':'not_hate'}])])]),
            html.Div([
                html.Div([dcc.Graph(id='histogram-1')],className='col-4'),
                html.Div([dcc.Graph(id='histogram-2')],className='col-4'),
                html.Div([dcc.Graph(id='histogram-3')],className='col-4')],className='row'),
            html.Br(),
            html.Div([
                html.Div([
                    html.H3('Wordcloud before cleaning text'),
                    html.Img(id='before_wc')],className='col-5'),
                html.Div([],className='col-1'),
                html.Div([
                    html.H3('Wordcloud after cleaning text'),
                    html.Img(id='after_wc')],className='col-4')],className='row'),
            html.Br(),
            html.Div([
                html.Div([
                    html.H3('Bigrams before cleaning text'),
                    html.Img(id='before_bigrams')],className='col-5'),
                html.Div([],className='col-1'),
                html.Div([
                    html.H3('Bigrams after cleaning text'),
                    html.Img(id='after_bigrams')],className='col-5')],className='row')]),
        dcc.Tab(value='tab-3', label='Prediction Model', children=[
            html.Br(),
            html.Div([
                html.P('Insert your comment:'),
                    dcc.Input(id='s_comment_text',
                            type='text',
                            value='',
                            size='400px')],className='col-9'),
            html.Br(),
            html.Button('Search', id='filter'),
            html.Br(),
            html.Br(),
            html.Div(id='predicted-values')])],className='row')],
    style={
        'maxWidth':'1200px',
        'margin': '0 auto'})

# Table callback
@app.callback(
    Output(component_id='div-table',component_property='children'),
    [Input(component_id='filter-df',component_property='n_clicks')],
    [State(component_id='filter-row',component_property='value'),
    State(component_id='filter-keyword',component_property='value'),
    State(component_id='filter-toxic',component_property='value'),
    State(component_id='filter-severe_toxic',component_property='value'),
    State(component_id='filter-obscene',component_property='value'),
    State(component_id='filter-threat',component_property='value'),
    State(component_id='filter-insult',component_property='value'),
    State(component_id='filter-ihate',component_property='value')])
def update_table(n_clicks,row,keyword,toxic,severe_toxic,obscene,threat,insult,ihate):
    df = pd.read_csv('train.csv')
    if keyword != '':
        df=df[df['comment_text'].str.contains(keyword)]
    if toxic != '':
        df=df[df['toxic']==toxic]
    if severe_toxic != '':
        df=df[df['severe_toxic']==severe_toxic]
    if obscene != '':
        df=df[df['obscene']==obscene]
    if threat != '':
        df=df[df['threat']==threat]
    if insult != '':
        df=df[df['insult']==insult]
    if ihate != '':
        df=df[df['identity_hate']==ihate]
    children=[generate_table(df,page_size=row)]
    return children

# Histogram 1
@app.callback(
    Output('histogram-1', 'figure'),
	[Input('filter-hate-type', 'value')])
def display_hist_1(filter_hate_type):
    df_eda1 = pd.read_csv('df_eda.csv')
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=df_eda1[df_eda1[filter_hate_type]==1]['exclamation'],nbinsx=15))
    fig1.update_layout(
        title="Histogram exclamation percentage in {} comments".format(filter_hate_type),
        xaxis_title="Exclamation Percentage",
        yaxis_title="Count",
        font=dict(
            size=9))
    return fig1

# Histogram 2
@app.callback(
    Output('histogram-2', 'figure'),
	[Input('filter-hate-type', 'value')])
def display_hist_2(filter_hate_type):
    df_eda2 = pd.read_csv('df_eda.csv')
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=df_eda2[df_eda2[filter_hate_type]==1]['capital_letter_percent'],nbinsx=15))
    fig2.update_layout(
        title="Histogram capital letter percentage in {} comments".format(filter_hate_type),
        xaxis_title="Capital Letter Percentage",
        yaxis_title="Count",
        font=dict(
            size=9))
    return fig2

# Histogram 3
@app.callback(
    Output('histogram-3', 'figure'),
	[Input('filter-hate-type', 'value')])
def display_hist_3(filter_hate_type):
    df_eda3 = pd.read_csv('df_eda.csv')
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df_eda3[df_eda3[filter_hate_type]==1]['capital_word_percent'],nbinsx=15))
    fig3.update_layout(
        title="Histogram capital word percentage in {} comments".format(filter_hate_type),
        xaxis_title="Capital Word Percentage",
        yaxis_title="Count",
        font=dict(
            size=9))
    return fig3

# Wordcloud_1
@app.callback(
    Output('before_wc', 'src'),
    [Input('filter-hate-type', 'value')])
def wordcloud_1(filter_hate_type):
    image_1 = 'before_{}.png'.format(filter_hate_type)
    encoded_image_1 = base64.b64encode(open(image_1, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_1.decode())

# Wordcloud_2
@app.callback(
    Output('after_wc', 'src'),
    [Input('filter-hate-type', 'value')])
def wordcloud_2(filter_hate_type):
    image_2 = 'after_{}.png'.format(filter_hate_type)
    encoded_image_2 = base64.b64encode(open(image_2, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_2.decode())

# Bigrams_1
@app.callback(
    Output('before_bigrams', 'src'),
    [Input('filter-hate-type', 'value')])
def bigrams_1(filter_hate_type):
    image_3 = 'bigrams_before_{}.png'.format(filter_hate_type)
    encoded_image_3 = base64.b64encode(open(image_3, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_3.decode())

# Bigrams_2
@app.callback(
    Output('after_bigrams', 'src'),
    [Input('filter-hate-type', 'value')])
def bigrams_2(filter_hate_type):
    image_4 = 'bigrams_after_{}.png'.format(filter_hate_type)
    encoded_image_4 = base64.b64encode(open(image_4, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_4.decode())

# Predictor callback
@app.callback(
    Output('predicted-values', 'children'),
	[Input('filter', 'n_clicks')],
    [State('s_comment_text', 'value')])
def predictor(n_clicks, comment_text):
    if comment_text == '':
        return "Please insert text"
    else:
        return try_model(comment_text)

if __name__ =='__main__':
    app.run_server(debug=True)