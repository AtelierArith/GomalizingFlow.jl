using ArgParse

using GomalizingFlow.Watcher

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "config"
        help = """
        specify path/to/a/toml/file
        you can find an example 'cfgs/example2d.toml'
        """
        required = true
        "--result"
        help = "path/to/result/dir"
        default = "result"
        "--item"
        help = "item to show during training e.g. ess, acceptance_rate"
        default = "ess"
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    path = args["config"]
    result = abspath(args["result"])
    item = Symbol(args["item"])
    result_dir = abspath(joinpath(result, splitext(basename(path))[begin]))
    @info "serving $(result_dir)"
    serve(result_dir, item)
end

main()
