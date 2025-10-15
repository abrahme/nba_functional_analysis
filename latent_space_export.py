import pandas as pd
import pickle
import re
import numpy as np
import argparse
from jax import config
config.update("jax_enable_x64", True)
import numpyro
import jax.numpy as jnp
from model.inference_utils import posterior_X_to_df


numpyro.set_platform("cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--cohort_year", help = "year format of {yyyy} indicating for which cohort dates prior and including will be included in training set", required=True, 
                        default = 2021, type = int)
    parser.add_argument("--player_names", help = "which players to run the model for", required=False, default = [], type = lambda x: x.split(","))

    args = vars(parser.parse_args())
    cohort_year = args["cohort_year"]
    players = args["player_names"]




    data = pd.read_csv("data/injury_player_cleaned.csv").query(f"age <= 38 & name != 'Brandon Williams'")
    data = data.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year
    names = data.groupby("id")["name"].first().values.tolist()
    id_df = data[["position_group","name","id", "minutes"]].groupby("id").max().reset_index()

    print("setup data")
    if players:
        pattern = r"class-of-(\d{4})"
        player_indices = []
        for item in players:
            match = re.fullmatch(pattern, item)
            if item == "low-minutes":
                total_mins = data.groupby("id")["minutes"].sum().reset_index()
                for index, val in enumerate(total_mins["minutes"].values.tolist()):
                    if val <= np.percentile(total_mins["minutes"], 25) and index not in player_indices: 
                        player_indices.append(index) 
            elif match:
                year = int(match.group(1))
                subset =  data.groupby("id")["year"].min().reset_index()
                
                for index,val in enumerate(subset["year"].values.tolist()):
                    if index not in player_indices and val == year:
                        player_indices.append(index)
            else:
                if names.index(item) not in player_indices:
                    player_indices.append(names.index(item))
    else:
        player_indices = []
    svi_paths = ["model_output/latent_max_boundary_convex_tvrflvm_map_negbin_cohort_2021_val_2022.pkl", "model_output/latent_max_boundary_convex_tvrflvm_map_negbin_cohort_2021_val_2023.pkl",
                 "model_output/latent_max_boundary_convex_tvrflvm_map_negbin_cohort_2021_val_2025.pkl"]
    mcmc_paths = ["model_output/latent_max_boundary_convex_tvrflvm_mcmc_negbin_cohort_2021_val_2022.pkl", "model_output/latent_max_boundary_convex_tvrflvm_mcmc_negbin_cohort_2021_val_2023.pkl",
                 "model_output/latent_max_boundary_convex_tvrflvm_mcmc_negbin_cohort_2021_val_2025.pkl"]

    X_dict = {}
    mu_dict = {}
    for svi_path, mcmc_path in zip(svi_paths, mcmc_paths):
        val_year = mcmc_path.split(".")[0][-4:]
        with open(svi_path, "rb") as f:
            results_map = pickle.load(f)
        f.close()
        results_map = {key.replace("__loc", ""): val for key,val in results_map.items()}
        with open(mcmc_path, "rb") as f:
            results_mcmc = pickle.load(f)
        f.close()
        results_collated = {**results_map, **results_mcmc}
        for item in results_collated:
            if item == "X":
                # print(len(player_indices), player_indices)
                # print("original", results_map["X"][jnp.array(player_indices)])
                X_new = jnp.tile(results_map["X"][None, None], (1, 50, 1, 1))
                # print("before", X_new[0,0, jnp.array(player_indices), :])
                X_new = X_new.at[..., jnp.array(player_indices), :].set(results_mcmc["X_free"])
                # print("item to set", results_mcmc["X_free"][0,0])
                # print("after", X_new[0,0 , jnp.array(player_indices), :])
                X_dict[val_year] = X_new
                results_collated["X"] = X_new
        
        
    
    dfs = []
    mu_dfs = []
    for val_year in X_dict:
        df = posterior_X_to_df(X_dict[val_year], id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], player_indices)
        df["val_year"] = val_year
        dfs.append(df)
        mu_df = pd.read_csv(f"posterior_mu_ar_{val_year}.csv")
        mu_df["val_year"] = val_year
        mu_dfs.append(mu_df)
    pd.concat(dfs).to_csv("latent_X_cohort_2022_2023_2025.csv", index = False)
    pd.concat(mu_dfs).to_csv("latent_mu_cohort_2022_2023_2025.csv", index = False)



