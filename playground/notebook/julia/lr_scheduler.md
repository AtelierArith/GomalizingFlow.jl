```julia
using Flux
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using Plots
```

```julia
schedule = Step(0.01, 0.1, [10, 10, 20])
epochs = 1:60
```

```julia
plot(epochs, [eta for (e, eta) in zip(epochs, schedule)])
```

```julia
opt = Scheduler(schedule, Momentum())
```

```julia
opt.optim
```
