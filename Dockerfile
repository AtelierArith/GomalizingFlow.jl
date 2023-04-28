# Usage: 
# docker build goma build-arg CUDA_VERSION=12.0.0
# 12.0.0 <- default configuration
# 11.7.0 <- you can also choose this
ARG CUDA_VERSION="12.0.0"

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    htop \
    nano \
    openssh-server \
    tig \
    tree \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install NodeJS
RUN apt-get update && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz && \
    mkdir "$JULIA_PATH" && \
    tar zxvf julia-1.8.5-linux-x86_64.tar.gz -C "$JULIA_PATH" --strip-components 1 && \
    rm julia-1.8.5-linux-x86_64.tar.gz # clean up

# Create user named jovyan which is compatible with Binder
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR /work/
RUN mkdir -p /work/
RUN chown -R ${NB_UID} /work/
USER ${NB_USER}

# Install basic packages on default environment
RUN julia -e 'using Pkg; Pkg.add(["PyCall", "IJulia", "Pluto", "PlutoUI", "Revise", "BenchmarkTools"]); Pkg.precompile()'

ENV PATH $PATH:${HOME}/.julia/conda/3/x86_64/bin

# Install packages for Jupyter Notebook/JupyterLab
RUN conda install -y -c conda-forge \
    jupyter \
    jupyterlab \
    jupytext \
    ipywidgets \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator \
    jupyter-server-proxy \
    nbconvert \
    ipykernel \
    jupyter-resource-usage \
    jupyterlab_code_formatter autopep8 black isort \
    && \
    conda clean -afy # clean up

# For Pluto.jl
RUN pip install \
    webio_jupyter_extension \
    webio-jupyterlab-provider \
    git+https://github.com/IllumiDesk/jupyter-pluto-proxy.git

# Install/enable extension for JupyterLab users
RUN jupyter labextension install jupyterlab-topbar-extension && \
    jupyter labextension install jupyterlab-system-monitor && \
    jupyter nbextension enable --py widgetsnbextension && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter labextension install @z-m-k/jupyterlab_sublime --no-build && \
    jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build && \
    jupyter serverextension enable --py jupyterlab_code_formatter && \
    jupyter labextension install @hokyjack/jupyterlab-monokai-plus --no-build && \
    jupyter labextension install @jupyterlab/server-proxy --no-build && \
    jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf ~/.cache/yarn && \
    rm -rf ~/.node-gyp && \
    echo Done

# Set color theme Monokai++ by default
RUN mkdir -p ${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    echo '{"theme": "Monokai++"}' >> \
    ${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

RUN mkdir -p ${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension && \
    echo '{"codeCellConfig": {"lineNumbers": true, "fontFamily": "JuliaMono"}}' \
    ${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

RUN mkdir -p ${HOME}/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension && \
    echo '{"shortcuts": [{"command": "runmenu:restart-and-run-all", "keys": ["Alt R"], "selector": "[data-jp-code-runner]"}]}' >> \
    ${HOME}/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/shortcuts.jupyterlab-settings

RUN wget https://raw.githubusercontent.com/mwouts/jupytext/main/binder/labconfig/default_setting_overrides.json -P  ~/.jupyter/labconfig/

RUN conda install -y seaborn matplotlib -c conda-forge && \
    conda install pytorch=1.12 torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
    conda clean -afy # clean up

# Install extra packages
RUN julia -e 'using Pkg; Pkg.add(["ImageFiltering", "WebIO", "Interact"])'

ENV JULIA_PROJECT=/work
USER root
COPY Project.toml /work/
RUN mkdir -p /work/src && echo "module GomalizingFlow end" > /work/src/GomalizingFlow.jl
RUN chown -R ${NB_UID} /work/Project.toml
USER ${NB_USER}

RUN julia -e '\
    using Pkg; Pkg.instantiate(); \
    Pkg.precompile(); \
    # Download CUDA artifacts \
    using CUDA, cuDNN; \
    using CUDA; CUDA.set_runtime_version!(v"${CUDA_VERSION}"); \
    '

RUN julia -e '\
    using CUDA; \
    if CUDA.functional() \
    @info "Downloading artifacts regarding CUDA and CUDNN for Julia"; \
    @assert CUDA.functional(true); \
    @assert cuDNN.has_cudnn(); \
    CUDA.versioninfo(); \
    end; \
    using InteractiveUtils; versioninfo() \
    '

# For Jupyter Notebook
EXPOSE 8888
# For Http Server
EXPOSE 8000
# For Pluto Server
EXPOSE 9999
ENV JULIA_EDITOR="code"
ENV EDITOR="nano"

RUN julia --threads auto -e 'using Base.Threads, IJulia; installkernel("julia", env=Dict("JULIA_NUM_THREADS"=>"$(nthreads())"))'
CMD ["julia"]
