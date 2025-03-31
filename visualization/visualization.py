import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jax.numpy as jnp
import plotly.express as px
import numpy as np
import arviz as az
from model.inference_utils import create_metric_trajectory, create_metric_trajectory_map

def plot_posterior_predictive_career_trajectory( player_index, metrics: list[str], metric_outputs: list[str], exposure_names: list[str],  posterior_mean_samples, observations, exposures, posterior_variance_samples):
    """
    plots the posterior predictive career trajectory 
    """
    fig = make_subplots(rows = 4, cols=4,  subplot_titles=metrics)
    
    observation_dict, posterior_dict = create_metric_trajectory(posterior_mean_samples, player_index,  observations, exposures, 
                                                                exposure_names=exposure_names,
                                                                metric_outputs=metric_outputs, metrics = metrics, posterior_variance_samples=posterior_variance_samples)

 
    obs = observation_dict["y"]
    posterior = posterior_dict["y"]
    x = list(range(18,39))
    for index, metric in enumerate(metrics):
        row = int(np.floor(index / 4)) + 1 
        col = (index % 4) + 1
        metric_type = metric_outputs[index]
        metric = metric.upper()
        if metric_type == "poisson":
            metric += " per 36 min"
        fig.add_trace(go.Scatter(x = x, y = obs[..., index], mode = "lines", 
                                 name = "Observed", line_color = "blue", showlegend=False), row = row, col=col)
        fig.add_trace(go.Scatter(x = x, y = jnp.nanmean(posterior[..., index], axis = (0,1)), mode = "lines", name = "Posterior Mean", line_color = "red", showlegend=False), row = row, col = col)
        hdi =  az.hdi(np.array(posterior[..., index]), hdi_prob = .95)
        fig.add_trace(go.Scatter(
        name='Upper Bound',
        x=x,
        y=hdi[:,1],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False),row = row, col=col)

        fig.add_trace(go.Scatter(
            name='Lower Bound',
            x=x,
            y=hdi[:,0],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False,
        ), row = row, col=col)

    fig.update_layout({'width':650, 'height': 650,
                            'showlegend':False, 'hovermode': 'closest',
                            })
    
    return fig


def plot_posterior_predictive_career_trajectory_map( player_index, metrics: list[str], metric_outputs: list[str], posterior_map, observations, exposures):
    """
    plots the posterior predictive career trajectory 
    """
    fig = make_subplots(rows = 4, cols=4,  subplot_titles=metrics)
    
    observation_dict, posterior_dict = create_metric_trajectory_map(posterior_map, player_index,  observations, exposures, 
                                                                metric_outputs=metric_outputs, metrics = metrics,)

 
    obs = observation_dict["y"]
    posterior = posterior_dict["y"]
    x = list(range(18,39))
    for index, metric in enumerate(metrics):
        row = int(np.floor(index / 4)) + 1 
        col = (index % 4) + 1
        metric_type = metric_outputs[index]
        metric = metric.upper()
        if metric_type == "poisson":
            metric += " per 36 min"
        fig.add_trace(go.Scatter(x = x, y = obs[..., index], mode = "lines", 
                                 name = "Observed", line_color = "blue", showlegend=False), row = row, col=col)
        fig.add_trace(go.Scatter(x = x, y = posterior[..., index], mode = "lines", name = "Posterior Mean", line_color = "red", showlegend=False), row = row, col = col)

    fig.update_layout({'width':650, 'height': 650,
                            'showlegend':False, 'hovermode': 'closest',
                            })
    
    return fig


def plot_mcmc_diagnostics(inference_data, variable_name, plot = "trace"):
    if plot == "trace":
        trace_plot = az.plot_trace(inference_data, var_names = variable_name)
        return trace_plot[0,1]
    elif plot == "post":
        return az.plot_posterior(inference_data, var_names = variable_name)[0,1]
    elif plot == "autocorr":
        return az.plot_autocorr(inference_data, var_names = variable_name)[0,1]
    elif plot == "summary":
        return az.summary(inference_data, var_names = variable_name)


def plot_correlation_dendrogram(X, labels, title = ""):
    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(X, orientation='bottom', labels=labels, )
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(X, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    heat_data = np.corrcoef(X)
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale="ice",
            reversescale=True
            
        )
    ]

    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout({'width':800, 'height':800,
                            'showlegend':False, 'hovermode': 'closest',
                            "title": title})
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'ticks':""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, .85],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks': ""
                            })
    # Edit yaxis2
    fig.update_layout(yaxis2={'domain':[.825, .975],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})
    
    
    return fig 

def plot_scatter(df, title= "", player_index:int = 0 ):
    fig = px.scatter_3d(df[~df.index.isin([player_index])], x = "dim1", y = "dim2", z = "dim3", color = "position_group", size = "minutes", opacity = .2, title=title,
                        hover_data = ["player_name", "minutes", "position_group"])
    fig.add_traces(px.scatter_3d(df[df.index.isin([player_index])], x = "dim1", y = "dim2", z = "dim3", size = "minutes", opacity = .9, title=title,
                        hover_data = ["player_name", "minutes", "position_group"]).data)
    return fig




        




            

