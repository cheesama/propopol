# -*- coding: utf-8 -*-
from tqdm import tqdm
from datetime import datetime, date, timedelta
from fbprophet import Prophet
from fastquant import backtest
from github import Github

from download_stock_data import get_all_stock_data

import pandas as pd
import os, sys
import time
import requests


def get_github_repo(access_token, repository_name):
    """
    github repo object를 얻는 함수
    :param access_token: Github access token
    :param repository_name: repo 이름
    :return: repo object
    """
    g = Github(access_token)
    repo = g.get_user().get_repo(repository_name)
    return repo


def upload_github_issue(repo, title, body):
    """
    해당 repo에 title 이름으로 issue를 생성하고, 내용을 body로 채우는 함수
    :param repo: repo 이름
    :param title: issue title
    :param body: issue body
    :return: None
    """
    repo.create_issue(title=title, body=body)


entire_df = get_all_stock_data()

# data frame title 변경 '회사명' = name, 종목코드 = 'code'
entire_df = entire_df.rename(columns={"Name": "name", "Code": "code", "Close": "y"})

target_folder = datetime.now().strftime("%Y_%m_%d") + "_stock_analysis"
os.makedirs(datetime.now().strftime("%Y_%m_%d") + "_stock_analysis", exist_ok=True)

from multiprocessing import Process, Pool

import multiprocessing

print("Number of cpu : ", multiprocessing.cpu_count())

# save all prediction result -> { code: prediction_low_bound - actual final close price}
predictions = {}
prediction_infos = {}

import plotly.offline as py
import plotly

# prediction args
min_period = 128
periods = 14
top_k = 10

# model training
start_time = time.time()
# last_date = (datetime.now()-timedelta(7)).strftime("%Y-%m-%d")
last_date = datetime.now() - timedelta(periods)

for corp_name in list(entire_df.name.unique()):
    df = entire_df[(entire_df["name"] == corp_name) & (entire_df["Market"] == "KOSPI")]
    df["ds"] = df.index

    # if program runs over 5h, break iteration
    if (time.time() - start_time) > 3600 * 5:
        break

    if len(df) < min_period:
        continue

    if df["ds"][-1] < last_date:
        continue

    try:
        name = df.iloc[0]["name"]
        code = df.iloc[0]["code"]

        # just focus common stock
        if "1신" in name or "1우" in name or "2신" in name:
            continue
        if name[-1] == "우":
            continue

        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        # fig = plot_plotly(m, forecast, xlabel=name + "(" + code + ")", figsize=(1200, 600))  # This returns a plotly Figure
        # fig.write_image(target_folder + os.sep + name + "(" + code + ").png")
        predictions[f"{name}({code})"] = (
            forecast.iloc[-1]["yhat_lower"] - df.iloc[-1]["y"]
        )
        prediction_infos[f"{name}({code})"] = {}
        prediction_infos[f"{name}({code})"]["current_price"] = df.iloc[-1]["y"]
        prediction_infos[f"{name}({code})"]["prediction_price"] = forecast.iloc[-1][
            "yhat_lower"
        ]
        prediction_infos[f"{name}({code})"]["expected_profit"] = (
            forecast.iloc[-1]["yhat_lower"] - df.iloc[-1]["y"]
        )

        print(
            f"\n{name}({code}) prediction finished!\texpected_profit : {forecast.iloc[-1]['yhat_lower'] - df.iloc[-1]['y']}"
        )

    except Exception as ex:
        print(ex)

# pick several positive corp prediction results
predictions = {
    k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)
}

# print top-k results
upload_contents = f"# {datetime.today().strftime('%Y-%m-%d')} stock_prediction(after {periods} days)\n\n"

# markdown table format set
upload_contents += (
    f"|   corp   |   current_price   |   prediction_price   |   expected_profit   |\n"
)
upload_contents += (
    f"|:--------:|:-----------------:|:--------------------:|:-------------------:|\n"
)

for i, (k, v) in enumerate(predictions.items()):
    if i > top_k:
        break
    print(
        f"corp: {k}\tcurrent_price:{prediction_infos[k]['current_price']}\tprediction_price:{prediction_infos[k]['prediction_price']}\texpected_profit: {prediction_infos[k]['expected_profit']}\n"
    )
    # upload_contents += f"corp: {k}\tcurrent_price:{prediction_infos[k]['current_price']}\tprediction_price:{prediction_infos[k]['prediction_price']}\texpected_profit: {prediction_infos[k]['expected_profit']}\n"
    upload_contents += f"|{k}|{prediction_infos[k]['current_price']}|{prediction_infos[k]['prediction_price']}|{prediction_infos[k]['expected_profit']}|\n"

os.environ["UPLOAD_CONTENTS"] = upload_contents

# generate result as github issue
issue_title = (
    f"{datetime.today().strftime('%Y-%m-%d')} stock_prediction(after {periods} days)"
)
access_token = os.environ["FULL_ACCESS_TOKEN"]
repository_name = "propopol"

repo = get_github_repo(access_token, repository_name)
upload_github_issue(repo, issue_title, upload_contents)
print("Upload Github Issue Success!")

# update readme for showing latest prediction result
with open("README.md", "w") as readmeFile:
    readmeFile.write(upload_contents)

# send result as slack webhook
webhook_url = os.environ["WEBHOOK_URL"]
webhook_payload = {"text": "Propopol Stock Predictor", "blocks": []}
for content in upload_contents.split("\n"):
    info_section = {"type": "section", "text": {"type": "mrkdwn", "text": f"{content}"}}
    webhook_payload["blocks"].append(info_section)
    webhook_payload["blocks"].append({"type": "divider"})

requests.post(url=webhook_url, json=webhook_payload)
