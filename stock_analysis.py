# -*- coding: utf-8 -*-
from tqdm import tqdm, notebook
from datetime import datetime, date
from fbprophet import Prophet

import pandas as pd
import pandas_datareader as pdr

from github import Github

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

# 종목 타입에 따라 download url이 다름. 종목코드 뒤에 .KS .KQ등이 입력되어야해서 Download Link 구분 필요
stock_type = {"kospi": "stockMkt", "kosdaq": "kosdaqMkt"}

# 회사명으로 주식 종목 코드를 획득할 수 있도록 하는 함수
def get_code(df, name):
    code = df.query("name=='{}'".format(name))["code"].to_string(index=False)
    # 위와같이 code명을 가져오면 앞에 공백이 붙어있는 상황이 발생하여 앞뒤로 sript() 하여 공백 제거
    code = code.strip()
    return code

# download url 조합
def get_download_stock(market_type=None):
    market_type = stock_type[market_type]
    download_link = "http://kind.krx.co.kr/corpgeneral/corpList.do"
    download_link = download_link + "?method=download"
    download_link = download_link + "&marketType=" + market_type
    df = pd.read_html(download_link, header=0)[0]
    return df


# kospi 종목코드 목록 다운로드
def get_download_kospi():
    df = get_download_stock("kospi")
    df.종목코드 = df.종목코드.map("{:06d}.KS".format)
    return df


# kosdaq 종목코드 목록 다운로드
def get_download_kosdaq():
    df = get_download_stock("kosdaq")
    df.종목코드 = df.종목코드.map("{:06d}.KQ".format)
    return df


# kospi, kosdaq 종목코드 각각 다운로드
kospi_df = get_download_kospi()
kosdaq_df = get_download_kosdaq()

# data frame merge
code_df = pd.concat([kospi_df, kosdaq_df])

# data frame정리
code_df = code_df[["회사명", "종목코드"]]

# data frame title 변경 '회사명' = name, 종목코드 = 'code'
code_df = code_df.rename(columns={"회사명": "name", "종목코드": "code"})

import os, sys

target_folder = datetime.now().strftime("%Y_%m_%d") + "_stock_analysis"
os.makedirs(datetime.now().strftime("%Y_%m_%d") + "_stock_analysis", exist_ok=True)

from multiprocessing import Process, Pool

import multiprocessing

print("Number of cpu : ", multiprocessing.cpu_count())
pool = Pool(multiprocessing.cpu_count())
# pool = Pool(1)

# save all prediction result -> { file_name: prediction_low_bound - actual final close price}
predictions = {}

from fbprophet.plot import plot_plotly

import plotly.offline as py
import plotly


#prediction args
start = datetime(2018, 1, 1)
end = datetime.date(datetime.now())
periods = 7
top_k = 10

#model training
for i in tqdm(range(len(code_df))):
    try:
        # get_data_yahoo API를 통해서 yahho finance의 주식 종목 데이터를 가져온다.
        df = pdr.get_data_yahoo(code_df.iloc[i]["code"], start, end).rename(
            columns={"Close": "y"}
        )
        df["ds"] = df.index

        name = code_df.iloc[i]["name"]
        code = code_df.iloc[i]["code"]

        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        fig = plot_plotly(
            m, forecast, xlabel=name + "(" + code + ")", figsize=(1200, 600)
        )  # This returns a plotly Figure
        # fig.show()
        fig.write_image(target_folder + os.sep + name + "(" + code + ").png")

        predictions[target_folder + os.sep + name + "(" + code + ").png"] = (
            forecast.iloc[-1]["yhat_lower"] - df.iloc[-1]["y"]
        )

        print(
            f"\n{name}({code}).png saved!\tvalue : {forecast.iloc[-1]['yhat_lower'] - df.iloc[-1]['y']}"
        )

    except Exception as ex:
        # print (ex)
        pass

# pick several positive corp prediction results
predictions = {
    k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)
}

# print top-k results
upload_contents = f"{datetime.today().strftime('%Y-%m-%d')} stock_prediction(after {periods} days)\n\n"
for i, (k, v) in enumerate(predictions.items()):
    if i > top_k:
        break
    print(f"corp: {k}, \t expected profit: {v}\n")
    upload_contents += f"corp: {k}, \t expected profit: {v}\n"


#generate result as github issue
issue_title = f"{datetime.today().strftime('%Y-%m-%d')} stock_prediction(after {periods} days)"
access_token = os.environ['GITHUB_TOKEN']
repository_name = "propopol"

repo = get_github_repo(access_token, repository_name)
upload_github_issue(repo, issue_title, upload_contents)
print("Upload Github Issue Success!")

