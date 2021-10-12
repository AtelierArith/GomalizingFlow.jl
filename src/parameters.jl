
struct DeviceParams
    device_id::Int
    device
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
    m²::Float32
    λ::Float32
end

function Base.getproperty(pp::PhysicalParams, s::Symbol)
    if s == :lattice_shape
        return ntuple(_ -> pp.L, pp.Nd)
    else
        return getfield(pp, s)
    end
end

@with_kw struct ModelParams
    n_layers::Int = 16
    hidden_sizes::Tuple = (8, 8)
    kernel_size::Int = 3
    inC::Int = 1
    outC::Int = 2
    use_final_tanh::Bool = true
end

@with_kw struct TrainingParams
    batchsize::Int = 64
    epochs::Int = 40
    iterations::Int = 100
    base_lr::Float32 = 0.001f0
    opt::String = "ADAM"
    prior = Normal{Float32}(0.f0, 1.f0)
end

struct HyperParams
    dp::DeviceParams
    tp::TrainingParams
    pp::PhysicalParams
    mp::ModelParams
end