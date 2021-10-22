using ArgParse

using LFT

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "config"
            help = """
            specify path/to/a/toml/file
            you can find an example 'cfgs/example2d.toml'
            """
            required = true
        "--device"
            help = "override Device ID"
            default = nothing
        "--result_root"
            help = "path/to/result/dir"
            default = "result"
        "--pretrained"
            help = "load /path/to/trained_model.bson and train with the model"
            default = nothing
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    path = args["config"]
    device_id = nothing
    if !isnothing(args["device"])
        device_id = parse(Int, args["device"])
    end
    pretrained = nothing
    if !isnothing(args["pretrained"])
        pretrained = parse(args["pretrained"])
    end
    result_root = abspath(args["result_root"])
    hp = LFT.load_hyperparams(path; device_id, pretrained)
    result_dir = joinpath(result_root, splitext(basename(hp.configpath))[begin])
    @info "create result dir $(result_dir)"
    mkpath(result_dir)
    @info "dump"
    LFT.hp2toml(hp, joinpath(result_dir, "config.toml"))

    trained_model = LFT.train(hp)
    
    @info "save model"
    LFT.BSON.@save joinpath(result_dir, "trained_model.bson") trained_model
    @info "make mcmc ensamble"
    nsamples = 8196
    history = make_mcmc_ensamble(model, prior, action, lattice_shape; batchsize, nsamples, device=cpu)
    LFT.BSON.@save joinpath(result_dir, "history.bson") history
end
