#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Output, Input, State
import plotly.express as px

import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import StructType,StructField, StringType
# test
from functools import reduce
import tweepy
import pandas as pd
import re
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--api-key', '--k', type=str, help='Twitter API key')
    return parser.parse_args()


def get_result(x):
    return x[0]


def get_tags(printer):
    tag_dict = {}
    for rule in printer.get_rules()[0]:
        tag_dict[rule[2]] = rule[0].replace(' -is:reply -is:retweet -has:links lang:en', '')
    return tag_dict



args = parse_args()
academic_bearer = "AAAAAAAAAAAAAAAAAAAAADIEawEAAAAAxzzD4cQ2g8FGK2%2BkKz2%2FJvTnoMA%3D09uegYs5HrQvrsFkAEl3WwxhspBYFBIH3Vnykec79asqiUsSoA"
streaming = tweepy.StreamingClient(academic_bearer)
spark = sparknlp.start()
pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang = "en")
RDD = spark.sparkContext.emptyRDD()
schema = StructType([
  StructField('text', StringType(), True),
  StructField('tags', StringType(), True)
  ])
df = spark.createDataFrame(RDD,schema)
app = dash.Dash('Twitter Sentiment Analysis')
app.layout = html.Div([
    html.H1('Twitter Sentiment Analysis'),
    html.Br(),
    html.H2(children='Input search terms'),
    html.Div([
        dcc.Input(
            id='input1',
            type='text',
            placeholder=None,
        ),
        dcc.Input(
            id='input2',
            type='text',
            placeholder=None,
        ),
        html.Button('Submit', id='submit-val', n_clicks=0),
    ]),
    html.Div(id='progress_status', children='Enter a value and press submit'),
    html.Div(id='hist', children=[]),
    dcc.Interval(
            id='interval_component',
            interval=10*1000, # in milliseconds
            n_intervals=0
    )
])

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

printer = None
thread = None


class IDPrinter(tweepy.StreamingClient):
    def __init__(self,academic_bearer):
        super().__init__(academic_bearer)

    def on_data(self, data):
        msg = json.loads( data )
        tags =[]
        #print(msg['data']['text'])
        tt = []
        text = msg['data']['text']
        global emoji_pattern
        tt.append(emoji_pattern.sub(r'', text))
        tag_dict = get_tags(self)
        for rule in msg['matching_rules']:
            tags.append(tag_dict[rule['id']])
        print(str("_".join(tags)))
        tt.append(str("_".join(tags)))
        global spark
        global schema
        global df
        newRow = spark.createDataFrame([tt], schema)
        df = df.union(newRow)


@app.callback(
    Output("hist", "children"),
    Input("interval_component", "n_intervals"),
    State("hist", "children"),
    prevent_initial_call=True
)
def show_graph(n_intervals, children):
    global printer
    tag_dict = get_tags(printer)
    print(printer.get_rules())
    print(tag_dict)
    sample = df.select(df.text, df.tags)[df.tags.isin(list(tag_dict.values()))]
    spark_result_df = pipeline.transform(sample).select('text', 'tags', 'sentiment.result')
    result_df = spark_result_df.withColumn('sentiment', get_result(spark_result_df.result)).toPandas()
    fig = px.histogram(result_df, x='sentiment', color='tags', category_orders={'sentiment': ['positive', 'negative', 'neutral']})
    fig.update_layout(barmode='group')
    if len(children) > 0:
        children[0]["props"]["figure"] = fig
    else:
        children.append(
            dcc.Graph(
                figure=fig
            )
        )

    return children

@app.callback(
    Output('progress_status', 'children'),
    State('input1', 'value'),
    State('input2', 'value'),
    Input("submit-val", "n_clicks"),
    prevent_initial_call=True
)
def set_tags(input1, input2, n_clicks):
    global printer
    global thread
    if printer:
        printer.disconnect()
    printer = IDPrinter(academic_bearer)
    printer.delete_rules([rule[2] for rule in printer.get_rules()[0]])
    printer.add_rules(tweepy.StreamRule(f"{input1} -is:reply -is:retweet -has:links lang:en"))
    printer.add_rules(tweepy.StreamRule(f"{input2} -is:reply -is:retweet -has:links lang:en"))
    thread = printer.filter(threaded=True)
    return f"Tags: {input1}, {input2}"

if __name__ == '__main__':
    app.run_server()
