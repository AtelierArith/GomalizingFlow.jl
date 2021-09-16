# LFT.jl

# Usage

## Setup environment (without Docker)

- Install jupyter and jupytext

```console
$ pip install jupyter jupytext
```

- install dependencies regarding julia

```console
$ julia -e 'ENV["PYTHON"]=Sys.which("python3"); ENV["JUPYTER"]=Sys.which("jupyter"); using Pkg; Pkg.add(["PyCall", "IJulia"])'
$ julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

- initialize jupyter server with the following command:

```console
$ cd path/to/this/repository
$ jupyter notebook
```

## Setup environment (with Docker)


### Case 1: GPU enabled

- If you want an environment with CUDA is enabled, please install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) by reading [this instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

- The following command will initialize jupyterlab

```console
$ make
$ docker-compose up lab-gpu # GPU
```

### Case 2: CPU

- If you're using macOS, you can't use CUDA. Please install [Install Docker Desktop on Mac](https://docs.docker.com/desktop/mac/install/). Then you're good to go.

```console
$ make
$ docker-compose up lab # CPU
```

- The following command will initialize jupyterlab
