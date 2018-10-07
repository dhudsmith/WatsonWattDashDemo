

#
#
# STARTER CODE
#
#
from flask import Flask, render_template, request, jsonify
import atexit
import os
import json


if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        # client = Cloudant(user, password, url=url, connect=True)
        # db = client.create_database(db_name, throw_on_exists=False)
# elif "CLOUDANT_URL" in os.environ:
#     client = Cloudant(os.environ['CLOUDANT_USERNAME'], os.environ['CLOUDANT_PASSWORD'], url=os.environ['CLOUDANT_URL'], connect=True)
#     db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        # url = 'https://' + creds['host']
        # client = Cloudant(user, password, url=url, connect=True)
        # db = client.create_database(db_name, throw_on_exists=False)

##
## Imports
import markdown_text

# Dash/plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Essentials
import pandas as pd
import numpy as np
from pprint import pprint
import logging
from matplotlib import cm

# set the logging level
logging.basicConfig(level=logging.INFO)

# Colors
from palettable.colorbrewer import qualitative as colors

def getColorScale(colorCode, colorrange, nbins=50):
    cmap = cm.get_cmap(colorCode)
    gradient = np.linspace(colorrange[0], colorrange[1], nbins)
    colors = ['rgb(%i, %i, %i)' % (x[0],x[1],x[2]) for x in cmap(gradient)[:,:3]*256]
    scale = [[x, y] for x, y in zip(gradient, colors)]

    return scale

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

colorInfo = {
    'sentiment': 'RdBu',
    'emotion': {
        'anger': {'data': 'watson_emotion_anger', 'color': 'Reds'},
        'disgust': {'data': 'watson_emotion_disgust', 'color': 'Greens'},
        'fear': {'data': 'watson_emotion_fear', 'color': 'Purples'},
        'joy': {'data': 'watson_emotion_joy', 'color': 'YlOrBr'},
        'sadness': {'data': 'watson_emotion_sadness', 'color': 'Blues'},
    }
}

default_camera = {'eye':
                      {'y': 0.53,
                       'x': 0.75,
                       'z': 0.7},
                  'up': {'y': 0,
                         'x': 0,
                         'z': 1},
                  'center': {'y': 0,
                             'x': 0,
                             'z': 0}
                  }

layout_3d = {}


##
## Helper functions:

def get_sentiment_string(row):
    sentiment_score = row.watson_sentiment
    sentiment = "Neutral"
    if sentiment_score >= 0.3:
        sentiment = "Positive"
    elif sentiment_score <= -0.3:
        sentiment = "Negative"

    sentiment_str = "<i>%s</i> (%.2f)" % (sentiment, sentiment_score)
    return (sentiment_str)


def get_top_emotion_string(row):
    emotions = row[['watson_emotion_anger', 'watson_emotion_disgust', 'watson_emotion_fear', 'watson_emotion_joy',
                    'watson_emotion_sadness']]
    max_emotion = max(emotions)
    emotion_str = 'None'
    if row.watson_emotion_anger == max_emotion:
        emotion_str = '<i>anger</i> (%.2f)' % max_emotion
    elif row.watson_emotion_disgust == max_emotion:
        emotion_str = '<i>disgust</i> (%.2f)' % max_emotion
    elif row.watson_emotion_fear == max_emotion:
        emotion_str = '<i>fear</i> (%.2f)' % max_emotion
    elif row.watson_emotion_joy == max_emotion:
        emotion_str = '<i>joy</i> (%.2f)' % max_emotion
    elif row.watson_emotion_sadness == max_emotion:
        emotion_str = '<i>sadness</i> (%.2f)' % max_emotion

    return (emotion_str)


##
## Data preparation

#  Prep the data
def configure_data(data):

    # TOPIC COLORS
    colorList = [colors.Dark2_8.colors,
                 colors.Paired_12.colors,
                 colors.Set2_8.colors,
                 colors.Set3_12.colors]

    colorList = ['rgb(' + str(item)[1:-1] + ')' for sublist in colorList for item in sublist]

    df_groups = data[['topic_code', 'cx', 'cy', 'cz']].groupby('topic_code').agg(np.mean).reset_index()
    df_groups['colors'] = df_groups.topic_code.apply(lambda x: colorList[x % len(colorList)])

    colorMap = [[x.topic_code / float(max(df_groups.topic_code)), x.colors] for _, x in df_groups.iterrows()]

    data = data.merge(df_groups.drop(['cx', 'cy', 'cz'], axis=1), on='topic_code')

    # ENRICHMENTS
    data['hovertext'] = data.apply(lambda x: x._text.replace('\n', '<br />') + \
                                               '<br><b>Watson Sentiment:</b> ' + \
                                               get_sentiment_string(x) + \
                                               '<br><b>Watson Top Emotion:</b> ' + \
                                               get_top_emotion_string(x),
                                     axis=1)

    return data, colorMap

def get_data_elements(data, colorMap):

    # get all of the data elements
    x, y, z, = np.asarray(data[['cx', 'cy', 'cz']]).transpose()
    topic_code = data['topic_code'].tolist()
    topic_colors = data['colors'].tolist()
    hovertext = np.asarray(data['hovertext']).transpose()
    sentiment = data['watson_sentiment'].tolist()
    anger, disgust, fear, joy, sadness = np.asarray(data[['watson_emotion_anger',
                                                          'watson_emotion_disgust',
                                                          'watson_emotion_fear',
                                                          'watson_emotion_joy',
                                                          'watson_emotion_sadness']]).transpose()


    elements = dict(
        coords = dict(x=x, y=y, z=z),
        text = hovertext,
        topic = dict(code=topic_code, color=topic_colors),
        watson = dict(sentiment=sentiment,
                      emotion=dict(anger=anger,
                                    disgust=disgust,
                                    fear=fear,
                                    joy=joy,
                                    sadness=sadness)
                      ),
        colorMap = colorMap
    )

    return elements


##
## Plot functions

# 3d scatterplot
def plot3dscatter(x, y, z, color_var, color_range, hover_text, label_colors, color_map, reversescale=False):

    # Create the plot
    plt = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            cmin=color_range[0],
            cmax=color_range[1],
            color=color_var,  # set color to an array/list of desired values
            colorscale=color_map,  # choose a colorscale
            reversescale=reversescale,
            opacity=0.8
        ),
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor=label_colors
        ),
        text=hover_text,
        textposition="top center"
    )

    return plt


# histogram
def plotHistogram(x, color_var, color_range, colorscale, nbins):
    plt = go.Histogram(x=x,
                       marker=dict(
                           cmin=color_range[0],
                           cmax=color_range[1],
                           color=color_var,
                           colorscale=colorscale,  # choose a colorscale
                           autocolorscale=False
                           ),
                       xbins=dict(
                           start=color_range[0],
                           end=color_range[1],
                           size=(color_range[1]-color_range[0])/float(nbins)
                           ),
                       autobinx=False
                       )

    return plt

##
## Build the 'figure' elements for the dashboard (plot + layout specification)

# 3D figures
def get3dfigure(dell, type, subtype='anger', camera=default_camera):
    # logic for choosing variables
    if type == 'topic':
        color_var = dell['topic']['code']
        color_range = [min(dell['topic']['code']), max(dell['topic']['code'])]
        label_colors = dell['topic']['color']
        color_map = dell['colorMap']
        reversescale=False
    elif type == 'sentiment':
        color_var = dell['watson']['sentiment']
        color_range = [-1,1]
        label_colors = 'aliceblue'
        color_map = colorInfo['sentiment']
        reversescale = True
    elif type == 'emotion':
        color_range = [0,1]
        color_bins=500
        label_colors = 'aliceblue'
        reversescale=False
        # different colors for different sub-types
        if subtype==1:
            color_var = dell['watson']['emotion']['anger']
            color_map = getColorScale(colorInfo['emotion']['anger']['color'], color_range, nbins=color_bins)
        elif subtype == 2:
            color_var = dell['watson']['emotion']['disgust']
            color_map = getColorScale(colorInfo['emotion']['disgust']['color'], color_range, nbins=color_bins)
        elif subtype == 3:
            color_var = dell['watson']['emotion']['fear']
            color_map = getColorScale(colorInfo['emotion']['fear']['color'], color_range, nbins=color_bins)
        elif subtype == 4:
            color_var = dell['watson']['emotion']['joy']
            color_map = getColorScale(colorInfo['emotion']['joy']['color'], color_range, nbins=color_bins)
        elif subtype == 5:
            color_var = dell['watson']['emotion']['sadness']
            color_map = getColorScale(colorInfo['emotion']['sadness']['color'], color_range, nbins=color_bins)
        else:
            logging.error("Invalid 3d plot sub-type: %s", subtype)
            raise Exception

    else:
        logging.error("Invalid 3d plot type: %s", type)
        raise Exception

    # build the figure
    fig = {
        'data': [plot3dscatter(x=dell['coords']['x'],
                               y=dell['coords']['y'],
                               z=dell['coords']['z'],
                               color_var = color_var,
                               color_range = color_range,
                               hover_text=dell['text'],
                               label_colors = label_colors,
                               color_map = color_map,
                               reversescale=reversescale)],
        'layout': go.Layout(
            height=900,
            margin={'l': 0, 'b': 10, 't': 10, 'r': 10},
            hovermode='closest',
            scene=dict(camera=camera)
        ),
    }

    return fig


# Histogram figure
def getHistFigure(dell, type, subtype=None):
    # logic for choosing plot characteristics based on type and subtype

    # SENTIMENT
    if type == 'sentiment':
        x = dell['watson']['sentiment']
        color_range = [-1, 1]
        colorscale = colorInfo['sentiment']
        nbins = 40
        reversescale = False
        xaxis = dict(
            tickvals=[-1,0, 1],
            range = [-1,1],
            ticktext=["Negative", "Neutral", "Positive"]
        )

    # EMOTION
    elif type == 'emotion':
        # settings for all emotion-type histograms
        color_range = [0, 1]
        nbins = 40
        xaxis = dict(
            tickvals=[0,1],
            range=[0,1],
            ticktext=["",""]
        )
        reversescale=True

        # sub-type differences
        if subtype ==1:
            x = dell['watson']['emotion']['anger']
            colorscale = getColorScale(colorInfo['emotion']['anger']['color'], color_range, nbins=nbins)
            xaxis['title']='Anger'
        elif subtype==2:
            x = dell['watson']['emotion']['disgust']
            colorscale = getColorScale(colorInfo['emotion']['disgust']['color'],color_range, nbins=nbins)
            xaxis['title'] = 'Disgust'
        elif subtype==3:
            x = dell['watson']['emotion']['fear']
            colorscale = getColorScale(colorInfo['emotion']['fear']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Fear'
        elif subtype==4:
            x = dell['watson']['emotion']['joy']
            colorscale = getColorScale(colorInfo['emotion']['joy']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Joy'
        elif subtype==5:
            x = dell['watson']['emotion']['sadness']
            colorscale = getColorScale(colorInfo['emotion']['sadness']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Sadness'
        else:
            logging.error("Invalid 3d plot sub-type: %s", subtype)
            raise Exception

    else:
        logging.error("Invalid 3d plot type: %s", type)
        raise Exception

    # Handle reversescale manually
    if reversescale:
        color_var = np.linspace(color_range[0], color_range[1], nbins)
    else:
        color_var = np.linspace(color_range[1], color_range[0], nbins)

    fig = {
        'data': [plotHistogram(x, color_var, color_range, colorscale, nbins)],
        'layout': go.Layout(
            height=200.,
            margin={'l': 50, 'b': 30, 't': 0, 'r': 25},
            dragmode='select',
            selectdirection='h',
            xaxis=xaxis,
            yaxis=dict(
                title='Count',
                type='log',
                autorange=True
            )
        )
    }

    return fig



def configure_dashboard(dell):

    # INITIALIZE THE DASHBOARD
    app = dash.Dash(__name__)

    # LAYOUT THE DASHBOARD -- main div
    app.layout = html.Div(children=[

        ##
        ## Title
        html.Div([
            dcc.Markdown(children=markdown_text.mrk_title),
            html.Hr(),
        ], style={'width': '100%', 'display': 'inline-block'}),

        ##
        ## Introduction
        html.Div([
            # left column
            html.Div([
                # 3d graph
                dcc.Graph(
                    id='scatter-3d-intro',
                    figure= get3dfigure(dell, 'topic')
                ),
            ], className='eight columns'),

            # right column
            html.Div([
                html.Div([
                    # section intro
                    dcc.Markdown(markdown_text.intro_title),
                    # description of graph
                    dcc.Markdown(markdown_text.intro_descr)
                ], id="introduction"),
            ], className='four columns'),

            # horizontal rule

        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Hr(),

        ##
        ## Sentiment
        html.Div([
            # left column
            html.Div([

                # text
                html.Div([
                    # section intro
                    dcc.Markdown(markdown_text.sentiment_title),
                    # description of graph
                    dcc.Markdown(markdown_text.sentiment_description)
                ], id="sentiment"),

                # histogram
                dcc.Graph(
                    id='hist-sentiment',
                    figure=getHistFigure(dell, 'sentiment')
                ),
            ], className='four columns'),

            #right column
            html.Div([

                # 3d graph
                dcc.Graph(
                    id='scatter-3d-sentiment',
                    figure=get3dfigure(dell, 'sentiment')
                ),
            ], className='eight columns'),

        ], style={'width': '100%', 'display': 'inline-block'}),
        # Horizontal rule
        html.Hr(),

        ##
        ## Emotion
        html.Div([

            # text at top
            html.Div([
                # section intro
                dcc.Markdown(markdown_text.emotion_title),
                # description of graph
                dcc.Markdown(markdown_text.emotion_description)
            ], id="emotion", style={"width": '60%', 'display': 'inline-block'}),

            # Columns
            html.Div([

                # left column
                html.Div([
                    # histograms
                    dcc.Graph(id='hist-emotion-anger', figure=getHistFigure(dell, 'emotion', 1)),
                    dcc.Graph(id='hist-emotion-disgust', figure=getHistFigure(dell, 'emotion', 2)),
                    dcc.Graph(id='hist-emotion-fear', figure=getHistFigure(dell, 'emotion', 3)),
                    dcc.Graph(id='hist-emotion-joy', figure=getHistFigure(dell, 'emotion', 4)),
                    dcc.Graph(id='hist-emotion-sadness', figure=getHistFigure(dell, 'emotion', 5)),
                ], className='four columns'),

                # right column
                html.Div([
                    # Color dropdown
                    html.Div([
                        dcc.Slider(
                            min=1,
                            max=5,
                            marks={
                                1: {'label': 'Anger', 'style': {'font-size':18}},
                                2: {'label': 'Disgust','style': {'font-size':18}},
                                3: {'label': 'Fear','style': {'font-size':18}},
                                4: {'label': 'Joy','style': {'font-size':18}},
                                5: {'label': 'Sadness','style': {'font-size':18}}
                            },
                            value=1, id='emotion-color-slider', included=False
                        ),
                    ], style={"width": '100%', 'height': '40px', 'display': 'inline-block', 'text-align': 'center'}),


                    # 3d graph
                    dcc.Graph(id='scatter-3d-emotion', figure=get3dfigure(dell, 'emotion', 1)),
                ], className='eight columns')
            ], style={"width": '100%', 'display': 'inline-block'})
            # Horisontal rule

        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Hr(),

    ], style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'})


    ##
    ## Callback function definitions

    # Sentiment graph callback definition:
    @app.callback(
        Output('scatter-3d-sentiment', 'figure'),
        [Input('hist-sentiment', 'selectedData')],
        [State('scatter-3d-sentiment', 'relayoutData')]
    )
    def plot3dFilteredSentiment(selectedData, relayoutData):

        # get the selected range
        try:
            range = selectedData["range"]["x"]
        except Exception as e:
            logging.info("Failed to get custom range or camera data for 3d sentiment plot. Error: %s", e)
            range = [-1., 1.]

        # get the current camera angle
        try:
            camera = relayoutData['scene.camera']
        except Exception as e:
            logging.info("Failed to get custom range or camera data for 3d sentiment plot. Error: %s", e)
            camera = default_camera

        # filter the data to the specified sentiment range
        sent = data['watson_sentiment'].between(range[0], range[1])
        data_f = data[sent]

        #  Prep the data
        dell = get_data_elements(data_f, colorMap)

        # return the figure
        return  get3dfigure(dell, 'sentiment', camera)

    # emotion color callback;
    @app.callback(
        Output('scatter-3d-emotion', 'figure'),
        [Input('hist-emotion-anger', 'selectedData'),
         Input('hist-emotion-disgust', 'selectedData'),
         Input('hist-emotion-fear', 'selectedData'),
         Input('hist-emotion-joy', 'selectedData'),
         Input('hist-emotion-sadness', 'selectedData'),
         Input('emotion-color-slider', 'value')],
        [State('scatter-3d-emotion', 'relayoutData')]
    )
    def plot3dFilteredSentiment(selAng, selDgst, selFear, selJoy, selSad, colorVal, relayoutData):

        # get the necc data from the input selections and color dropdown
        rangeAng = selAng["range"]["x"] if selAng is not None else [0,1]
        rangeDgst = selDgst["range"]["x"] if selDgst is not None else [0,1]
        rangeFear = selFear["range"]["x"] if selFear is not None else [0,1]
        rangeJoy = selJoy["range"]["x"] if selJoy is not None else [0,1]
        rangeSad = selSad["range"]["x"] if selSad is not None else [0,1]

        # camera is a little more tricky
        camera = default_camera
        if relayoutData is not None:
            if 'scene.camera' in relayoutData:
                camera = relayoutData['scene.camera']

        # filter the data
        isAng = data['watson_emotion_anger'].between(rangeAng[0], rangeAng[1])
        isDgst = data['watson_emotion_disgust'].between(rangeDgst[0], rangeDgst[1])
        isFear = data['watson_emotion_fear'].between(rangeFear[0], rangeFear[1])
        isJoy = data['watson_emotion_joy'].between(rangeJoy[0], rangeJoy[1])
        isSad = data['watson_emotion_sadness'].between(rangeSad[0], rangeSad[1])
        data_f = data[isAng & isDgst & isFear & isJoy & isSad]

        #  Prep the data
        dell = get_data_elements(data_f, colorMap)

        # return the figure
        return get3dfigure(dell, 'emotion', colorVal, camera)


    return app


if __name__ == '__main__':
    ##
    ## Globally define data elements
    dataPath = 'fb_posts_tsne_3d.csv'
    data = pd.read_csv(dataPath, encoding='utf-8', sep=',')

    # prepare the data
    data, colorMap = configure_data(data)
    data_elements = get_data_elements(data, colorMap)

    # build the dashboard
    app = configure_dashboard(data_elements)

    # run the server
    app.run_server(host='0.0.0.0', port=8080, debug=True, processes=4)