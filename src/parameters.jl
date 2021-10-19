
struct DeviceParams
    device_id::Int
    device::Function # expected Flux.cpu or Flux.gpu
    function DeviceParams(device_id)
        if device_id >= 0 && CUDA.functional()
            CUDA.device!(device_id)
            device = gpu
            @info "Training on GPU with device_id=$(device_id)"
        else
            if device_id > 0
                @warn "You've set device_id = $device_id, but CUDA.functional() is $(CUDA.functional())"
            end
            @info "Training on CPU"
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
    result::String = "result"
end

struct HyperParams
    dp::DeviceParams
    tp::TrainingParams
    pp::PhysicalParams
    mp::ModelParams
    path::String
end

function load_hyperparams(path::AbstractString; override_device_id::Union{Nothing,Int}=nothing)::HyperParams
    toml = TOML.parsefile(path)
    if !isnothing(override_device_id)
        @info "override device id $(override_device_id)"
        device_id = override_device_id
    else
        device_id = toml["device_id"]
    end
    dp = DeviceParams(device_id)
    tp = ToStruct.tostruct(TrainingParams, toml["training"])
    pp = ToStruct.tostruct(PhysicalParams, toml["physical"])
    mp = ToStruct.tostruct(ModelParams, toml["model"])
    return HyperParams(dp, tp, pp, mp, path)
end
