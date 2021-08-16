# LFT.jl

## Usage

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

