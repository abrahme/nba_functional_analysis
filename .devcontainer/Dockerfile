FROM nvcr.io/nvidia/jax:24.04-py3

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
        file \
        curl \
        g++ \
        tmux \
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
        # "jax[cuda12]" \ 
        numpyro \ 
        arviz \
        optax \
        shiny \
        plotly \
        shinywidgets \
        Jinja2 

## GitHub CLI Installation
# RUN wget https://github.com/cli/cli/releases/download/v${GITHUB_CLI_VERSION}/gh_${GITHUB_CLI_VERSION}_linux_amd64.tar.gz -O - | \
#     tar xvzf - -C /opt/conda/bin gh_${GITHUB_CLI_VERSION}_linux_amd64/bin/gh --strip-components=2

