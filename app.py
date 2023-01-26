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
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, split

from functools import reduce
import tweepy
import pandas as pd
import re
import json
from datetime import datetime
import time
from confluent_kafka import Producer
import socket

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
spark = SparkSession.builder.appName('twitter_app')\
    .master("local[*]")\
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1,com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.8')\
    .config('spark.streaming.stopGracefullyOnShutdown', 'true')\
    .config("spark.driver.memory","8G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .getOrCreate()
pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang = "en")
app = dash.Dash('Twitter Sentiment Analysis')
app.layout = html.Div([
    html.H1('Nastawienie uzytkownikow Twittera'),
    html.Br(),
    html.H2(children='Wpisz tagi do porownania'),
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
        html.Button('Zapisz', id='submit-val', n_clicks=0),
    ]),
    html.Div(id='progress_status', children='Wpisz tagi i kliknij Zapisz'),
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
topic = 'tt'
schema = StructType([
  StructField('text', StringType(), True),
  StructField('tags', StringType(), True)
  ])
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", topic) \
    .load()


class IDPrinter(tweepy.StreamingClient):
    def __init__(self,academic_bearer):
        super().__init__(academic_bearer)
        conf = {'bootstrap.servers': "localhost:9092",
        'client.id': socket.gethostname()}

        self.producer = Producer(conf)

    def on_data(self, data):
        try:
            msg = json.loads( data )
            tags =[]
            #print(msg['data']['text'])
            tt = {}
            text = msg['data']['text']
            global emoji_pattern
            tt['text'] = (emoji_pattern.sub(r'', text))
            tag_dict = get_tags(self)
            for rule in msg['matching_rules']:
                tags.append(tag_dict[rule['id']])
            print(str("_".join(tags)))
            tt['tags'] = (str("_".join(tags)))
            global topic
            self.producer.produce(topic, key=str(time.time()).encode(), value=json.dumps(tt))
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")


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
    input1 = list(tag_dict.values())[0]
    input2 = list(tag_dict.values())[1]
    text_df = spark.read.format('parquet').load(f"tmp/output_dir").select('key', 'text', 'tags').filter(f"tags in ('{input1}', '{input2}')")
    spark_result_df = pipeline.transform(text_df).select('key', 'text', 'tags', 'sentiment.result')
    result_df = spark_result_df.withColumn('sentiment', get_result(spark_result_df.result)).toPandas()
    fig = px.histogram(
        result_df,
        x='sentiment',
        color='tags',
        category_orders={'sentiment': ['positive', 'negative', 'neutral']},
        color_discrete_map={ # replaces default color mapping by value
                input1: "#1f77b4", input2: "#d62728"
            },
    )
    fig.update_layout(barmode='group')
    fig.update_layout(
        title="Suma pozytywnych, negatywnych i neutralnych wpisow",
        xaxis_title='Tag',
        yaxis_title='Suma'
    )
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
    df_select1 = df.select((from_json(col("value").cast("string"), schema)).alias('text'), col('topic'), col('key').cast('string'))
    text_df = df_select1.select('key',
                          col('text.tags').alias('tags'),
                          col('text.text').alias('text'),
                          'topic')
    text_df.writeStream \
        .format("parquet") \
        .option("checkpointLocation", f"tmp/checkpoint") \
        .option("path", f"tmp/output_dir") \
        .start()
    return f"Tagi: {input1}, {input2}"


if __name__ == '__main__':
    app.run_server()
