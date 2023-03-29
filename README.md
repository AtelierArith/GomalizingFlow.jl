# GomalizingFlow.jl: A Julia package for Flow-based sampling algorithm

# Abstract

GomalizingFlow.jl is a package to generate configurations using the flow based sampling algorithm in Julia programming language.
This software serves two main purposes: to acceralate research works of lattice QCD with machine learning with easy prototyping, and to provide an independent implimentation to an existing code in Python/PyTorch.
GomalizingFlow.jl implements,
the flow based sampling algorithm, namely, RealNVP and Metropolis-Hastings test for two dimension and three dimensional scalar field, which can be switched by a parameter file.
HMC for that theory also implemented for comparison.
This code works not only on CPU but also on NVIDIA GPU.

# Usage (TL; DR)

```console
$ git clone https://github.com/AtelierArith/GomalizingFlow.jl && cd GomalizingFlow.jl
$ # Install Docker and GNU Make command:
$ make
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml
```

# Usage (detailed description)

If you're familiar with how to use Julia especially deep learning with GPU, you can setup an environment by yourself via:

```julia
julia> using Pkg; Pkg.instantiate()
julia> using GomalizingFlow
julia> hp = GomalizingFlow.load_hyperparams("cfgs/example2d.toml"; device_id=0, pretrained=nothing, result="result")
julia> GomalizingFlow.train(hp)
```

Otherwise, we recommend to create one using Docker container (see the following instructions from Step1 to Step4).

## Step1: Setup environment (using Docker)

[Install Docker, more precisely NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), [Docker Compose](https://docs.docker.com/compose/install/compose-plugin/#installing-compose-on-linux-systems) and GNU Make.

Below shows author's development environment (Linux/Ubuntu 20.04).

```console
$ cat /etc/lsb-release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.2 LTS"
$ make  make --version
GNU Make 4.3
Built for x86_64-pc-linux-gnu
Copyright (C) 1988-2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
$ docker --version
Docker version 23.0.1, build a5ee5b1
$ docker-compose --version
docker-compose version 1.29.2, build 5becea4c
$ nvidia-smi
Fri Mar 24 23:34:16 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   33C    P8     9W / 280W |      6MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|  0%   36C    P8    11W / 280W |    179MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
$ cat /etc/docker/daemon.json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

You can install Docker Compose easily via `pip3 install docker-compose`.

## Step2: Build a Docker image

Just do:

```console
$ make
```

It will create a Docker image named `gomalizingflowjl`.

## Step3: Start training (using Docker)

### Syntax

In general, you can train a model via:

```console
$ docker-compose run --rm julia julia begin_training.jl <path/to/config.toml>
```

For example:

```console
$ docker-compose run --rm julia julia begin_training.jl cfgs/example2d.toml # You can run in a realistic time without using GPU accelerator.
```

It will generate some artifacts in `results/example2d/`:

```console
$ ls result/example2d/
Manifest.toml               evaluations.csv             src
Project.toml                history.bson                trained_model.bson
config.toml                 history_best_ess.bson       trained_model_best_ess.bson
```

You'll see a trained model (`trained_model.bson`), a training log (`evaluations.csv`) or configurations generated by flow based sampling algorithm (`history.bson`).

For those who are interested in training for 3D field theory. Just run:

```console
$ docker-compose run --rm julia julia begin_training.jl cfgs/example3d.toml # For 3D field theory parameter, you may want to use GPU for training.
```

### Watch `evaluations.csv` during training

During training, you can watch the value of ess for each epoch.

- Run training script as usual:

```console
$ docker-compose run --rm julia julia begin_training.jl </path/to/config.toml>
```

- Open another terminal, and run the following:

```console
$ docker-compose run --rm julia julia watch.jl </path/to/config.toml>
```

It will display a plot in your terminal something like:

```console
$ docker-compose run --rm julia julia watch.jl cfgs/example2d.toml
Creating gomalizingflowjl_julia_run ... done
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

We also support showing `acceptance_rate` by adding `--item acceptance_rate`:

```console
$ docker-compose run --rm julia julia watch.jl cfgs/example2d_E1.toml --item acceptance_rate
Creating gomalizingflowjl_julia_run ... done
[ Info: serving /work/result/example2d_E1
[ Info: evaluations.csv is updated
                      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Evaluation⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                      ┌────────────────────────────────────────────────────────────┐
                   60 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ acceptance_rate
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⣀⣀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣄⠀⢀⣀⢰⠒⢢⡠⠤⠜⠈⣧⠃⠈⠊⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⡀⡰⠓⢄⠖⠒⠒⠚⠀⠙⡼⢸⢸⠀⠀⠁⠀⠀⠀⠘⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⢀⣠⠋⠚⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠁⠈⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⡤⠔⠜⠈⠁⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢄⣰⠙⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   acceptance_rate    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠴⠒⠉⠞⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⢤⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⠀⠀⠀⠀⠀⡠⡄⡔⠚⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠀⠀⡠⠔⠉⠉⠉⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      │⠈⠓⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                    0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
                      └────────────────────────────────────────────────────────────┘
                      ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀90⠀
                      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

## Step4: Analyse results

After training in Step3, you can load a trained model you specify and calculate zero-momentum
two point functions.

```julia
using CUDA, Flux, ParameterSchedulers # require to call restore function
using GomalizingFlow, Plots, ProgressMeter
backend = unicodeplots() # select backend
# You can also call `gr()`
# backend = gr()

r = "result/example2d"
# r = "result/example3d"

trained_model, history = GomalizingFlow.restore(r);

hp = GomalizingFlow.load_hyperparams(joinpath(r, "config.toml"))

lattice_shape = hp.pp.lattice_shape
cfgs = Flux.MLUtils.batch(history[:x][2000:end]);
T = eltype(cfgs)

y_values = T[]
@showprogress for t in 0:hp.pp.L
    y = GomalizingFlow.mfGc(cfgs, t)
    push!(y_values, y)
end
plot(0:hp.pp.L, y_values, label="AffineCouplingLayer")
```

The result should be:

```julia
julia> plot(0:hp.pp.L, y_values, label="AffineCouplingLayer")
               ┌────────────────────────────────────────┐
     0.0366151 │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀│ AffineCouplingLayer
               │⠀⡿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀│
               │⠀⡇⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀│
               │⠀⡇⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀│
               │⠀⡇⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀│
               │⠀⡇⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀│
               │⠀⡇⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
               │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   0.000588591 │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠑⠒⠢⠤⠒⠒⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
               └────────────────────────────────────────┘
               ⠀-0.24⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀8.24⠀
```

See `playground/notebook/julia/analysis_tool.md` or `playground/notebook/julia/hmc.md` to get more examples.

# Directory structure

Below is the directory structure of our project:

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

## Playground

There are lots of jupyter notebooks in `playground/notebook/julia` regarding our project. Readers can learn about the trial and error process that led to the release of the software. You can run Jupyter Lab server locally as usual via:

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

# Citation

Bibtex item can be obtained from [iNSPIRE-HEP](https://inspirehep.net/literature/2138819):

```
@article{Tomiya:2022meu,
    author = "Tomiya, Akio and Terasaki, Satoshi",
    title = "{GomalizingFlow.jl: A Julia package for Flow-based sampling algorithm for lattice field theory}",
    eprint = "2208.08903",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "8",
    year = "2022"
}
```

# Appendix

- [Gomalizing...?](https://twitter.com/AkioTomiya/status/1560623282981707781?s=20)
  - You need a sense of humor to understand it.
- [Combinational-convolution for flow-based sampling algorithm](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_31.pdf) See:
  - Code is available: [GomalizingFlow.jl/playground/notebook/julia/4d_flow_4C3.md](https://github.com/AtelierArith/GomalizingFlow.jl/blob/combi-conv/playground/notebook/julia/4d_flow_4C3.md) in [combi-conv](https://github.com/AtelierArith/GomalizingFlow.jl/tree/combi-conv) tag.
