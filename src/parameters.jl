
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
    use_bn::Bool = false
end

@with_kw struct TrainingParams
    seed::Int = 12345
    batchsize::Int = 64
    epochs::Int = 40
    iterations::Int = 100
    base_lr::Float64 = 0.001
    opt::String = "Adam"
    prior::String = "Normal{Float32}(0.f0, 1.f0)"
    lr_scheduler::String = ""
    pretrained::String = ""
end

struct HyperParams
    configversion::VersionNumber
    dp::DeviceParams
    tp::TrainingParams
    pp::PhysicalParams
    mp::ModelParams
    result_dir::String
end

function load_hyperparams(
    config::Dict,
    output_dirname::String;
    device_id::Union{Nothing,Int}=nothing,
    pretrained::Union{Nothing,String}=nothing,
    result::AbstractString="result",
)::HyperParams
    configversion = VersionNumber(string(config["config"]["version"]))
    if !isnothing(device_id)
        @info "override device id $(device_id)"
    else
        device_id = config["device"]["device_id"]
    end

    if !isnothing(pretrained)
        @info "restore model from $(pretrained)"
        config["training"]["pretrained"] = pretrained
    end

    dp = DeviceParams(device_id)
    tp = ToStruct.tostruct(TrainingParams, config["training"])
    pp = ToStruct.tostruct(PhysicalParams, config["physical"])
    if !("use_bn" in keys(config["model"]))
        config["model"]["use_bn"] = false
    end
    mp = ToStruct.tostruct(ModelParams, config["model"])
    result_dir = abspath(joinpath(result, output_dirname))
    return HyperParams(configversion, dp, tp, pp, mp, result_dir)
end

function _d(configpath::AbstractString)
    foldername = splitext(basename(configpath))[begin]
    return foldername
end

function load_hyperparams(
    configpath::AbstractString,
    output_dirname::String=_d(configpath),
    args...;
    kwargs...,
)
    config = TOML.parsefile(configpath)
    load_hyperparams(config, output_dirname, args...; kwargs...)
end

function hp2toml(hp::HyperParams, fname::AbstractString)
    data = OrderedDict{String,Any}()
    data["config"] = OrderedDict{String,Any}("version" => string(hp.configversion))
    data["device"] = OrderedDict{String,Any}("device_id" => hp.dp.device_id)
    for (sym, itemname) in [(:mp, "model"), (:pp, "physical"), (:tp, "training")]
        obj = getfield(hp, sym)
        v = OrderedDict(key => getfield(obj, key) for key in fieldnames(obj |> typeof))
        data[itemname] = v
    end
    open(fname, "w") do io
        TOML.print(io, data)
    end
end
