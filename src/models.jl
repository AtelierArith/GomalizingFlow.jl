struct AffineCoupling{Net,Mask}
    net::Net
    mask::Mask
end

Flux.@functor AffineCoupling

function (model::AffineCoupling)(x_pair_loghidden)
    x = x_pair_loghidden[begin]
    loghidden = x_pair_loghidden[end]
    x_frozen = model.mask .* x
    x_active = (1 .- model.mask) .* x
    # (inW, inH, inD, inB) -> (inW, inH, inD, 1, inB) # by Flux.unsqueeze(*, 4)
    net_out = model.net(Flux.unsqueeze(x_frozen, ndims(x_frozen)))
    s = @view net_out[.., 1, :] # extract feature from 1st channel
    t = @view net_out[.., 2, :] # extract feature from 2nd channel
    fx = @. (1 - model.mask) * t + x_active * exp(s) + x_frozen
    logJ = sum((1 .- model.mask) .* s, dims=1:(ndims(s) - 1))
    return (fx, loghidden .- logJ)
end

# alias
forward(model::AffineCoupling, x_pair_loghidden) = model(x_pair_loghidden)

function create_model(hp::HyperParams)
    # device configurations
    device = hp.dp.device
    # physical configurations
    lattice_shape = hp.pp.lattice_shape
    # network configurations
    seed = hp.mp.seed
    Random.seed!(seed)
    n_layers = hp.mp.n_layers
    hidden_sizes = hp.mp.hidden_sizes
    kernel_size = hp.mp.kernel_size
    if kernel_size isa Int
        kernel_size = ntuple(_ -> kernel_size, hp.pp.Nd)
    end
    inC = hp.mp.inC
    outC = hp.mp.outC
    use_final_tanh = hp.mp.use_final_tanh

    module_list = []
    for i ∈ 0:(n_layers - 1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        net = []
        for (c, c_next) ∈ pairwise(channels)
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            k = 1 / (c * prod(kernel_size))
            W = rand(Uniform(-√k, √k), kernel_size..., c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            push!(net, mycircular)
            push!(net, Conv(W, b, leakyrelu, pad=0))
        end
        if use_final_tanh
            c = channels[end - 1]
            c_next = channels[end]
            k = 1 / (c * prod(kernel_size))
            W = rand(Uniform(-√k, √k), kernel_size..., c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            net[end] = Conv(W, b, tanh, pad=0)
        end
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list, coupling)
    end
    model = Chain(module_list...) |> f32 |> device
    return model
end

function get_training_params(model)
    ps = Flux.params(model)
    for i in 1:length(model)
        delete!(ps, model[i].mask)
    end
    return ps
end