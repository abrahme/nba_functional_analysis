# This file generated by Quarto; do not edit by hand.
# shiny_mode: core

from __future__ import annotations

from pathlib import Path
from shiny import App, Inputs, Outputs, Session, ui

import pandas as pd
import numpy as np
from shiny.express import input, render, ui
from shinywidgets import render_plotly

# ========================================================================

from data.data_utils import create_fda_data
data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
names = data.groupby("id")["name"].first().values
metric_output = ["binomial", "exponential"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
data["retirement"] = 1
data["log_min"] = np.log(data["minutes"])
data["simple_exposure"] = 1 
_ , outputs, _ = create_fda_data(data, basis_dims=3, metric_output=metric_output, 
                                     metrics = metrics
, exposure_list =  exposure_list)
observations = np.stack([output["output_data"] for output in outputs], axis = 1)
exposures = np.stack([output["exposure_data"] for output in outputs], axis = 0)

# ========================================================================




def server(input: Inputs, output: Outputs, session: Session) -> None:
    from visualization.visualization import plot_career_trajectory_observations
    ui.input_select(id="player", label = "Select a player", choices = {index : name for index, name in enumerate(names)})
    @render_plotly
    def plot_metric_arc():
        return plot_career_trajectory_observations(int(input.player()), metrics, metric_output, observations, exposures )

    # ========================================================================



    return None


_static_assets = ["advancement_files","images/noun-deep-learning-1705425.png","images/noun-scatter-graph-4768711.png"]
_static_assets = {"/" + sa: Path(__file__).parent / sa for sa in _static_assets}

app = App(
    Path(__file__).parent / "advancement.html",
    server,
    static_assets=_static_assets,
)
