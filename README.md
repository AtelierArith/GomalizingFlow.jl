# GomalizingFlow.jl

# Usage (今北産業, TL; DR)


```console
$ # Install Docker and GNU Make command:
$ make
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```

# Usage (detailed description)

## Setup environment (without using Docker)

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

## Setup environment (using Docker)

We would like to add an example of hardware/software environment

```console
$ docker --version
Docker version 20.10.12, build e91ed57
$ docker-compose --version
docker-compose version 1.29.1, build c34c88b2
$ nvidia-smi
Sun Mar  6 00:30:45 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   29C    P8     9W / 280W |      6MiB / 11178MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|  0%   30C    P8     9W / 280W |     15MiB / 11177MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
$ cat /etc/docker/daemon.json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

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


## Start training (without using Docker)

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
$ julia --project=@. begin_training.jl cfgs/example2d.toml
$ julia --project=@. begin_training.jl cfgs/example2d.toml --device=1 # train with GPU 1
```

- as for 3D lattice:

```julia
$ julia --project=@. begin_training.jl cfgs/example3d.toml
```

After training, we'll find `result/<config.toml>/trained_model.bson` is created. You can resotre the file in another Julia session something like this:

```julia
julia> using BSON: @load
julia> @load "path/to/trained_model.bson" trained_model
julia> # do something
```

See https://github.com/JuliaIO/BSON.jl for more information.

## Start training (using Docker)

### Case 1: You have a CUDA-Enabled machine

```julia
$ docker-compose run --rm julia-gpu julia begin_training.jl cfgs/example3d.toml
$ # equivalently
$ docker run --gpus all --rm -it -v $PWD:/work -w /work GomalizingFlowjl julia -e 'using Pkg; Pkg.instantiate()'
$ docker run --gpus all --rm -it -v $PWD:/work -w /work GomalizingFlowjl julia begin_training.jl cfgs/example2d.toml
```

### Case 2: CPU only

```julia
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```

#### M1 mac users

```console
$ docker build -t GomalizingFlowjl -f docker/Dockerfile.m1 .
$ docker run --rm -it -v $PWD:/work -w /work GomalizingFlowjl julia --project=/work -e 'using Pkg; Pkg.instantiate()'
$ docker run --rm -it -v $PWD:/work -w /work GomalizingFlowjl julia begin_training.jl cfgs/example2d.toml
```

## Optional

- You can watch the value of `ess` during training

### Usage

- Run training script as usual:

```console
$ docker-compose run --rm julia julia begin_training.jl /path/to/config.toml
```

- Open another terminal, then run the following:

```console
$ docker-compose run --rm julia julia watch.jl /path/to/config.toml
```

It will display a plot something like:

```
[ Info: evaluations.csv is updated
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Evaluation⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
           ┌────────────────────────────────────────────────────────────┐
       0.2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ ess
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⢠⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢸⠀⠀⠀⠀⡜⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠈⡆⠀⠀⢀⠇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⢆⠀⠀⠀⡇⠀⠀⢣⠀⠀⡸⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   ess     │⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠈⢢⠀⢸⠀⠀⠀⠸⡀⠀⡇⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⢸⠀⠀⠀⠀⢠⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠣⡇⠀⠀⠀⠀⢇⢸⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⡇⠀⠀⠀⡎⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⢸⠀⠀⡸⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⡇⢠⠃⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⢸⡎⠀⠀⠀⠀⠘⠤⢄⣀⣀⡠⠔⠒⠒⠊⠑⠤⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           └────────────────────────────────────────────────────────────┘
           ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀20⠀
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```
