FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up

# Install NodeJS
RUN apt-get update && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz && \
    mkdir "$JULIA_PATH" && \
    tar zxvf julia-1.6.2-linux-x86_64.tar.gz -C "$JULIA_PATH" --strip-components 1 && \
    rm julia-1.6.2-linux-x86_64.tar.gz # clean up

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
RUN julia -e 'using Pkg; Pkg.add(["PyCall", "IJulia", "Pluto", "PlutoUI"]); Pkg.precompile()'

ENV PATH $PATH:${HOME}/.julia/conda/3/bin

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
    jupyterlab_code_formatter autopep8 black \
    && \
    conda clean -afy # clean up

# For Pluto.jl
RUN pip install git+https://github.com/IllumiDesk/jupyter-pluto-proxy.git

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
    echo '{"codeCellConfig": {"lineNumbers": true}}' \
    >> ${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

ENV JULIA_PROJECT=/work
USER root
COPY ./Project.toml $JULIA_PROJECT/Project.toml
RUN chown -R ${NB_UID} /work/Project.toml
USER ${NB_USER}

RUN julia -e '\
    using Pkg; Pkg.instantiate(); \
    Pkg.precompile(); \
    # Download CUDA artifacts \
    using CUDA; \
    if CUDA.functional() \
        # Download artifacts of CUDA/CUDNN
        @assert CUDA.functional(true); \
        @assert CUDA.has_cudnn(); \
    end; \
    using InteractiveUtils; versioninfo() \
'

#RUN conda install -c anaconda cudatoolkit
RUN conda install -y matplotlib pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia && \
    conda clean -afy # clean up
    

# For Jupyter Notebook
EXPOSE 8888
# For Http Server
EXPOSE 8000
# For Pluto Server
EXPOSE 9999

CMD ["julia"]
