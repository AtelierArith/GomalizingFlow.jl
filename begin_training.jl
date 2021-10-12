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
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    path = args["config"]
    override_device_id = nothing
    if !isnothing(args["device"])
        override_device_id = parse(Int, args["device"])
    end
    hp = LFT.load_hyperparams(path; override_device_id)
    LFT.train(hp)
end
