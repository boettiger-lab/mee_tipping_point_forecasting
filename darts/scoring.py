import properscoring as ps
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--file",
    default="block_rnn_saddle_1000.csv.gz",
    type=str,
    help="file from forecasts to score",
)
args = parser.parse_args()

df = pd.read_csv(f"forecasts/{args.file}", index_col=0)
scores = []
for i in range(5):
    iter_df = df[df["iter"] == i]
    emp_df = iter_df[iter_df["true_model"] == True]
    forecast_df = iter_df[iter_df["true_model"] == False]
    iter_scores = []
    
    for t in emp_df["time"].unique():
        observation = emp_df[emp_df["time"] == t]["value"].to_numpy()
        forecasts = forecast_df[forecast_df["time"] == t]["value"].to_numpy()
        for forecast in forecasts:
            score = ps.crps_ensemble(forecast, observation)
            iter_scores.append(score)
    scores.append(iter_scores)
scores = np.array(scores).flatten().mean()
print(score)
