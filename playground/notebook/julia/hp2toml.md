---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Julia-sys 1.6.3
    language: julia
    name: julia-sys-1.6
---

```julia
using TOML
using GomalizingFlow
```

```julia
configpath = joinpath(dirname(dirname(pathof(GomalizingFlow))), "cfgs", "example2d.toml")
```

```julia
String(read(configpath)) |> print
```

```julia
config = TOML.parsefile(configpath)
```

```julia
hp = GomalizingFlow.load_hyperparams(configpath)
```

```julia
using DataStructures
```

```julia
function hp2toml(hp)
    tomldata = OrderedDict{String, Any}("device_id" => hp.dp.device_id)
    for (sym, itemname) in [(:mp, "model"), (:pp, "physical"), (:tp, "training")]
        obj = getfield(hp, sym)
        v = OrderedDict(key=>getfield(obj, key) for key ∈ fieldnames(obj |> typeof))
        tomldata[itemname] = v
    end
    tomldata

    TOML.print(tomldata)
end

hp2toml(hp)
```
