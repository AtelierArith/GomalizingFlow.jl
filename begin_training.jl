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
        pretrained = parse(Int, args["pretrained"])
    end
    hp = LFT.load_hyperparams(path; device_id, pretrained)
    LFT.train(hp)
end
