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
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    path = args["config"]
    hp = LFT.load_hyperparams(path)
    LFT.train(hp)
end


