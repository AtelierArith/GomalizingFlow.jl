
struct DeviceParams
    device_id::Int
    device::Function # expected Flux.cpu or Flux.gpu
    function DeviceParams(device_id)
        if device_id >= 0 && CUDA.functional()
            CUDA.device!(device_id)
            device = gpu
            @info "Set device: GPU with device_id=$(device_id)"
        else
            if device_id > 0
                @warn "You've set device_id = $device_id, but CUDA.functional() is $(CUDA.functional())"
            end
            @info "Set device: CPU"
            device_id = -1
            device = cpu
        end
        new(device_id, device)
    end
end

@with_kw struct PhysicalParams
    L::Int
    Nd::Int
    M2::Float64
    lam::Float64
end

function Base.getproperty(pp::PhysicalParams, s::Symbol)
    s == :lattice_shape && return ntuple(_ -> pp.L, pp.Nd)
    s == :m² && return getfield(pp, :M2)
    s == :λ && return getfield(pp, :lam)
    return getfield(pp, s)
end

@with_kw struct ModelParams
    seed::Int = 2021
    n_layers::Int = 16
    hidden_sizes::Vector{Int} = [8, 8]
    kernel_size::Int = 3
    inC::Int = 1
    outC::Int = 2
    use_final_tanh::Bool = true
end

@with_kw struct TrainingParams
    seed::Int = 12345
    batchsize::Int = 64
    epochs::Int = 40
    iterations::Int = 100
    base_lr::Float64 = 0.001
    opt::String = "ADAM"
    prior::String = "Normal{Float32}(0.f0, 1.f0)"
    pretrained::String = ""
    result::String = "result"
end

struct HyperParams
    dp::DeviceParams
    tp::TrainingParams
    pp::PhysicalParams
    mp::ModelParams
    configpath::String
end

function load_hyperparams(
        configpath::AbstractString;
        device_id::Union{Nothing,Int}=nothing,
        pretrained::Union{Nothing,String}=nothing,
    )::HyperParams
    config = TOML.parsefile(configpath)
    if !isnothing(device_id)
        @info "override device id $(device_id)"
    else
        device_id = config["device_id"]
    end

    if !isnothing(pretrained)
        @info "restore model from $(pretrained)"
        config["training"]["pretrained"] = pretrained
    end

    dp = DeviceParams(device_id)
    tp = ToStruct.tostruct(TrainingParams, config["training"])
    pp = ToStruct.tostruct(PhysicalParams, config["physical"])
    mp = ToStruct.tostruct(ModelParams, config["model"])
    return HyperParams(dp, tp, pp, mp, configpath)
end

function hp2toml(hp::HyperParams, fname::AbstractString)
    data = OrderedDict{String, Any}("device_id" => hp.dp.device_id)
    for (sym, itemname) in [(:mp, "model"), (:pp, "physical"), (:tp, "training")]
        obj = getfield(hp, sym)
        v = OrderedDict(key=>getfield(obj, key) for key ∈ fieldnames(obj |> typeof))
        data[itemname] = v
    end
    open(fname, "w") do io
        TOML.print(io, data)
    end
end