module Watcher

using FileWatching
using UnicodePlots
using CSV, DataFrames

export serve

function serve(dir, item::Symbol, title = "Evaluation")
    file = joinpath(dir, "evaluations.csv")
    while true
        FileWatching.watch_file(file)
        sleep(0.5)
        @info "$(basename(file)) is updated"
        try
            df = CSV.read(file, DataFrame)
            p = lineplot(
                df.epoch, getproperty(df, item);
                name = string(item),
                title = title,
                xlabel = "epoch",
                ylabel = string(item),
                width = 60
            )
            display(p)
            println()
        catch
            @warn "something happened"
        end
    end
end

end