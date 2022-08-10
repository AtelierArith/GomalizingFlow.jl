# GomalizingFlow.jl

# Usage (TL; DR)

```console
$ git clone https://github.com/AtelierArith/GomalizingFlow.jl && cd GomalizingFlow.jl
$ # Install Docker and GNU Make command:
$ make
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```

# Usage (detailed description)

If you're familiar with how to use Julia especially machine learning or GPU programming, you can setup environment by yourself via:

```julia
julia> using Pkg; Pkg.instantiate()
julia> using GomalizingFlow
julia> hp = GomalizingFlow.load_hyperparams("cfgs/example2d.toml"; device_id=0, pretrained=nothing, result="result")
julia> GomalizingFlow.train(hp)
```

Otherwise, we recommend to create one using Docker container.

## Setup environment (using Docker)

[Install Docker, more precisely NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), [Docker Compose](https://docs.docker.com/compose/install/compose-plugin/#installing-compose-on-linux-systems) and GNU Make.

Below shows author's development environment (Linux/Ubuntu 20.04).

```console
$ make --version
GNU Make 4.2.1
Built for x86_64-pc-linux-gnu
$ docker --version
Docker version 20.10.17, build 100c701
$ docker-compose --version
docker-compose version 1.29.2, build 5becea4c
$ nvidia-smi
Thu Aug 11 02:17:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   35C    P8     9W / 280W |      6MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|  0%   34C    P8    13W / 280W |     15MiB / 11264MiB |      0%      Default |
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

## Build a Docker image

Just do

```console
$ make
```

## Start training (using Docker)

### Syntax

In general, you can train a model via:

```console
$ docker-compose run --rm julia julia begin_training.jl <path/to/config.toml>
```
For example:

```julia
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml # You can enjoy our software on CPU-only hardware device
```

It will generate trained model in `result/example2d/`:

```console
$ ls result/example2d/
Manifest.toml               evaluations.csv             src
Project.toml                history.bson                trained_model.bson
config.toml                 history_best_ess.bson       trained_model_best_ess.bson
```

For those who are interested in training for a 3D lattice. Just run:

```julia
$ docker-compose run --rm julia julia begin_training.jl cfgs/example3d.toml # For 3D lattice case, you may want to use GPU for training.
```

### Visualize ess.

During training, you can watch the value of ess for each epoch.

- Run training script as usual:

```console
$ docker-compose run --rm julia julia begin_training.jl /path/to/config.toml
```

- Open another terminal, and run the following:

```console
$ docker-compose run --rm julia julia watch.jl /path/to/config.toml
```

It will display a plot in your terminal something like:

```console
$ docker-compose run --rm julia julia watch.jl cfgs/example2d.toml
[ Info: serving ~/work/atelier_arith/GomalizingFlow.jl/result/example2d
[ Info: evaluations.csv is updated
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Evaluation⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
           ┌────────────────────────────────────────────────────────────┐
       0.8 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ ess
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡄⠀⠀⠀⠀⠀⠀⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⢀⠀⠀⠀⡀⣴⠀⠀⠀⢸⡇⠀⣷⣄⣀⠀⢀⠀⡄│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⣿⠀⢠⠀⣼⠀⡇⣸⡇⣿⠀⠀⠀⢸⣷⢰⣿⡇⣿⢀⣸⣰⡇│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⡆⡀⠀⢀⣴⡄⠀⡇⣴⡄⣿⡀⣾⠀⣿⣧⢳⣿⣧⣿⡀⠀⠀⣼⣿⣿⡏⠃⣿⣾⣿⣿⡇│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⣴⡇⠀⡄⡇⣧⠀⢸⡏⡇⠀⡇⣿⡇⡿⡇⣿⠀⣿⣿⢸⣿⣿⣿⢿⡇⠀⡏⢻⡿⠁⠀⣿⡏⣿⣿⣷│
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢰⡀⢀⡆⣴⣿⢣⠀⣧⣧⣿⣦⠿⠃⢸⠀⣷⢹⣧⡇⢣⣿⡄⣿⣿⠘⡇⣿⣿⢸⡇⡞⠃⢸⡇⠀⠀⣿⡇⣿⢿⣿│
   ess     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⣸⡇⣿⣿⡏⣿⢸⣠⢻⡿⠉⢻⠀⠀⢸⣼⣿⢸⣿⡇⢸⣿⢻⢻⣿⠀⠀⢹⡟⠸⣷⡇⠀⢸⡇⠀⠀⣿⡇⣿⠀⡿│
           │⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠀⣇⠀⠀⠀⢀⢸⢸⣿⣇⡇⢹⡇⣿⠀⣿⠸⠇⠀⢸⠀⠀⢸⣿⣿⢸⡿⠀⠈⠋⢸⢸⣿⠀⠀⢸⡇⠀⠸⡇⠀⠸⡇⠀⠀⣿⠁⡏⠀⠁│
           │⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⢰⣿⡀⢀⡀⡸⣾⢸⣿⣿⡇⢸⡇⡿⠀⣿⠀⠀⠀⠘⠀⠀⢸⡏⣿⢸⡇⠀⠀⠀⢸⢸⡟⠀⠀⢸⡇⠀⠀⡇⠀⠀⡇⠀⠀⣿⠀⡇⠀⠀│
           │⠀⠀⠀⠀⠀⠀⠀⠀⡸⡄⣾⣿⣇⣼⣇⡇⢿⢸⢿⣿⡇⢸⠀⡇⠀⣿⠀⠀⠀⠀⠀⠀⠈⠁⣿⠀⠁⠀⠀⠀⠘⠈⠃⠀⠀⢸⠁⠀⠀⠀⠀⠀⠃⠀⠀⣿⠀⡇⠀⠀│
           │⠀⠀⡄⡀⠀⠀⣧⠀⡇⣧⣿⣿⣿⣿⣿⡇⠘⠘⢸⣿⡇⠸⠀⡇⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⡇⠀⠀│
           │⡀⠀⣧⣇⢸⠀⣿⢸⠁⠀⡇⠻⣿⠛⠻⠇⠀⠀⠀⣿⡇⠀⠀⠀⠀⠿⠀⠀⠀⠀⠀⠀⠀⠀⠙⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⡇⠀⠀│
           │⢣⣶⣿⢹⡜⡄⡏⡾⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠀⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀│
         0 │⠈⠸⠉⠀⠇⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
           └────────────────────────────────────────────────────────────┘
           ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀200⠀
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

# Directory structure

Below is our directory structure

```console
$ tree -d
.
├── cfgs
├── playground # The process of trial and error is documented.
│   ├── notebook
│   │   ├── julia
│   │   └── python
│   └── pluto
├── src # contains training script etc...
└── test # `make test` runs the package's test/runtests.jl file 
    ├── assets
    └── pymod
```

# Playground

There are lots of notebooks in `playground/notebook/julia` regarding our program. The manuscript here is a draft. Readers can learn about the trial and error process that led to the release of the software. You can run Jupyter Lab server locally as usual via:

```console
$ docker-compose up lab
Creating gomalizingflowjl-lab ... done
Attaching to gomalizingflowjl-lab
# Some stuff happen
gomalizingflowjl-lab |
gomalizingflowjl-lab |     To access the server, open this file in a browser:
gomalizingflowjl-lab |         file:///home/jovyan/.local/share/jupyter/runtime/jpserver-1-open.html
gomalizingflowjl-lab |     Or copy and paste one of these URLs:
gomalizingflowjl-lab |         http://gomagomakyukkyu:8888/lab?token=xxxxxxxxxx
gomalizingflowjl-lab |      or http://127.0.0.1:8888/lab?token=xxxxxxxxxx # Click this link in your terminal
```

We track Jupyter Notebooks as `.md` with the power of [jupytext](https://github.com/mwouts/jupytext) rather than `.ipynb`.
