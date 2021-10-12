# LFT.jl

# Usage (今北産業, TL; DR)


```console
$ # Install Docker and GNU Make command:
$ make
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```

# Usage (detailed description)

## Setup environment (without Docker)

- Install jupyter and jupytext

```console
$ pip install numpy matplotlib torch jupyter jupytext
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


### Case 1: You have a CUDA-Enabled machine

- If you want a Docker environment with CUDA, please install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) by reading [this instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

- The following command will initialize jupyterlab

```console
$ make
$ docker-compose up lab-gpu # GPU
```

### Case 2: CPU only

- If you're using macOS, for example, you can't use CUDA. Please install [Install Docker Desktop on Mac](https://docs.docker.com/desktop/mac/install/). Then you're good to go.
- The following command will initialize jupyterlab

```console
$ make
$ docker-compose up lab # CPU
```


## Start training (without Docker)

### syntax

In general:

```julia
julia --project=@. begin_training.jl path/to/config.toml
```

- The `path/to/config.toml` above has a option named `device_id` which accepts an integer >= -1. 
  - If you set `device_id = 0`. Our software is trying to use GPU its Device ID is `0`.
  - Setting `device_id = -1` will train model on CPU
- Optionally, you can override a Device ID by setting the `--device=<device_id>`

### examples

for example as for 2D lattice:

```julia
julia --project=@. begin_training.jl cfgs/example2d.toml
julia --project=@. begin_training.jl cfgs/example2d.toml --device=1 # train with GPU 1
```

- as for 3D lattice:

```julia
julia --project=@. begin_training.jl cfgs/example3d.toml
```

After training, `result/<config.toml>/trained_model.bson` is created. You can resotre the file on Julia session something like this:

```julia
julia> using BSON: @load
julia> @load "path/to/trained_model.bson" trained_model
julia> # do something
```

See https://github.com/JuliaIO/BSON.jl for more information.

## Start training (with Docker)

### Case 1: You have a CUDA-Enabled machine

```julia
docker-compose run --rm julia-gpu julia begin_training.jl cfgs/example3d.toml
```

### Case 2: CPU only

```julia
docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```
