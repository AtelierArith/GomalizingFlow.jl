# Usage:
# docker build goma build-arg CUDA_VERSION=12.0.0
# 12.0.0 <- default configuration
# 11.7.0 <- you can also choose this
ARG CUDA_VERSION="11.7.0"

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

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.2-linux-x86_64.tar.gz && \
    mkdir "$JULIA_PATH" && \
    tar zxvf julia-1.10.2-linux-x86_64.tar.gz -C "$JULIA_PATH" --strip-components 1 && \
    rm julia-1.10.2-linux-x86_64.tar.gz # clean up

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

ENV PATH=$PATH:$HOME/.rye/shims
RUN curl -sSf https://rye-up.com/get | RYE_VERSION="0.32.0" RYE_INSTALL_OPTION="--yes" bash
RUN $HOME/.rye/shims/rye config --set-bool behavior.use-uv=true

RUN $HOME/.rye/shims/rye tools install jupyterlab \
    && $HOME/.rye/shims/rye tools install jupytext \
    && $HOME/.rye/shims/rye tools install ruff

RUN mkdir -p ${HOME}/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension && \
    echo '{"shortcuts": [{"command": "runmenu:restart-and-run-all", "keys": ["Alt R"], "selector": "[data-jp-code-runner]"}]}' >> \
    ${HOME}/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/shortcuts.jupyterlab-settings

RUN jupytext-config set-default-viewer

# Install basic packages for default environment
RUN julia -e 'using Pkg; ENV["PYTHON"]=expanduser("~/.rye/shims/python3"); Pkg.add(["IJulia", "PyCall", "Pluto", "PlutoUI", "Revise", "BenchmarkTools"]); Pkg.precompile()'
# Install extra packages
RUN julia -e 'using Pkg; Pkg.add(["ImageFiltering"])'

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
    env_cuda_version = ENV["CUDA_VERSION"]; \
    using CUDA; CUDA.set_runtime_version!(VersionNumber(env_cuda_version)); \
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

RUN julia -e 'using Pkg; Pkg.instantiate()'

# For Jupyter Notebook
EXPOSE 8888
# For Http Server
EXPOSE 8000
# For Pluto Server
EXPOSE 9999
ENV JULIA_EDITOR="code"
ENV EDITOR="nano"

RUN julia --threads auto -e 'using Base.Threads, IJulia; installkernel("Julia", "--project=@.", env=Dict("JULIA_NUM_THREADS"=>"$(nthreads())"))'
CMD ["julia"]
