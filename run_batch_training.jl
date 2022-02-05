using TOML

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
        "--result"
        help = "path/to/result/dir"
        default = "result"
        "--pretrained"
        help = "load /path/to/trained_model.bson and train with the model"
        default = nothing
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    path = args["config"]
    device_id = nothing
    if !isnothing(args["device"])
        device_id = parse(Int, args["device"])
    end
    pretrained = nothing
    if !isnothing(args["pretrained"])
        pretrained = abspath(args["pretrained"])
    end
    result = abspath(args["result"])
    config = TOML.parsefile(path)
    for L in [4, 8, 12, 16]
        config["physical"]["L"] = L
        foldername = splitext(basename(configpath))[begin]
        foldername * "L_$(L)"
        @info "foldername" foldername
        hp = LFT.load_hyperparams(config, foldername; device_id, pretrained, result)
        LFT.train(hp)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
