FROM docker.io/rocker/tidyverse:latest
RUN install2.r --error --deps TRUE uwot HDInterval ggrepel ggridges ggnewscale pheatmap gt ggdist nnTensor ggbeeswarm umap ggforce dbscan

### Environment variables
ENV GITHUB_CLI_VERSION 2.30.0

###########################
### SYSTEM INSTALLATION ###
###########################
USER root

### System dependencies. Feel free to add packages as necessary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Basic system usage
        lmodern \
        dialog \
        file \
        curl \
        g++ \
        tmux \
        curl \
        tcl8.6 \
        tk8.6 \
        libtcl8.6 \
        libtk8.6 \
        ###################################################
        ### Add your own system dependencies installed  ###
        ### with `apt-get` as needed below this comment ###
        ### Example (note the backslash after name):    ###
        ### neofetch \                                  ###
        ###################################################
        && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts


## GitHub CLI Installation
RUN (type -p wget >/dev/null || ( apt update &&  apt-get install wget -y)) \
&&  mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg |  tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&&  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |  tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&&  apt update \
&&  apt install gh -y