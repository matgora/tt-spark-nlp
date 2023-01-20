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
    html.H2(children='Input search term'),
    html.Div([
        dcc.Input(
            id='tt_input',
            type='text',
            placeholder=None,
        ),
        html.Button('Submit', id='submit-val', n_clicks=0),
    ]),
    html.Div(id='progress_status', children='Enter a value and press submit'),
    html.Div(id='hist', children=[]),
])

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

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
        for rule in msg['matching_rules']:
          if(rule['id'] == "1616159160550178822"):
            tags.append('biden')
          else:
            tags.append('trump')
        # print(str("_".join(tags)))
        tt.append(str("_".join(tags)))
        global spark
        global schema
        global df
        newRow = spark.createDataFrame([tt], schema)
        df = df.union(newRow)


printer = IDPrinter(academic_bearer)
printer.add_rules(tweepy.StreamRule("trump -is:reply -is:retweet -has:links lang:en"))
printer.add_rules(tweepy.StreamRule("biden -is:reply -is:retweet -has:links lang:en"))
printer.filter(threaded=True)
@app.callback(
    Output("hist", "children"),
    Input("submit-val", "n_clicks"),
    State("hist", "children"),
    prevent_initial_call=True
)
def show_graph(state, children):
    # data = spark.read.format("csv").load('training.1600000.processed.noemoticon.csv')
    # old_columns = data.schema.names
    # new_columns = ["target", "id", "date", "flag", "user", "text"]
    # spark_df = reduce(lambda data, idx: data.withColumnRenamed(old_columns[idx], new_columns[idx]), range(len(old_columns)), data)
    # sample = spark_df.sample(False, 0.1, seed=0)
    sample = df.select(df.text, df.tags)
    spark_result_df = pipeline.transform(sample).select('text', 'tags', 'sentiment.result')
    result_df = spark_result_df.withColumn('result_flatten', get_result(spark_result_df.result)).toPandas()

    fig = px.histogram(result_df, x='result_flatten', color='tags')
    if len(children) > 0:
        children[0]["props"]["figure"] = fig
    else:
        children.append(
            dcc.Graph(
                figure=fig
            )
        )

    return children


if __name__ == '__main__':
    app.run_server()