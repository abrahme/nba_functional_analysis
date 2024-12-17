FROM nvcr.io/nvidia/jax:24.10-py3

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
        pandoc \
        pandoc-citeproc \
        curl \
        gdebi-core \
        ###################################################
        ### Add your own system dependencies installed  ###
        ### with `apt-get` as needed below this comment ###
        ### Example (note the backslash after name):    ###
        ### neofetch \                                  ###
        ###################################################
        && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts

### PyPI (pip) packages
RUN pip install \ 
        ################################################
        ### Add your own PyPI dependencies installed ###
        ### with `pip` as needed below this comment  ###
        ### Example (note the backslash after name): ###
        ### scikit-ntk \                             ###
        ################################################
        numpyro \ 
        blackjax \
        arviz \
        optax \
        shiny \
        plotly \
        shinywidgets \
        Jinja2 \
        scikit-learn \
        kaleido

# RUN pip install --upgrade "jax[cuda12]"

## GitHub CLI Installation
RUN (type -p wget >/dev/null || ( apt update &&  apt-get install wget -y)) \
&&  mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg |  tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&&  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |  tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&&  apt update \
&&  apt install gh -y


# Quarto installation
RUN curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb
RUN gdebi --non-interactive quarto-linux-amd64.deb