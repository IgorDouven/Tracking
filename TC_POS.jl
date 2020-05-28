using HypothesisTests
using PlotlyJS
using Distributed
addprocs()
@everywhere begin
    using Distributions
    using DataFrames
    using StatsBase
    using RCall
end

@everywhere R"""
suppressMessages(suppressWarnings(library(ROCR)))
suppressMessages(suppressWarnings(library(BayesFactor)))
"""

@everywhere begin
    dif(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64}) = sum(v[X ∩ Z]) / sum(v[Z]) - sum(v[X])
    
    finch(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64}) = (sum(v[X ∩ Z]) / sum(v[Z])) / sum(v[X]) - 1
    
    function deltp(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notZ = setdiff(collect(1:length(v)), Z)
        sum(v[X ∩ Z]) / sum(v[Z]) - sum(v[X ∩ notZ]) / sum(v[notZ])
    end

    function carn(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
      sum(v[X ∩ Z]) - sum(v[Z]) * sum(v[X])
    end

    function ctg_0(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notX = setdiff(collect(1:length(v)), X)
        rel = dif(v, X, Z) / sum(v[notX])
        irr = dif(v, X, Z) / sum(v[X])
        return dif(v, X, Z) >= 0 ? rel : irr
    end

    function ctg(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64}, α::Float64)
        dif(v, X, Z) >= 0 ? ctg_0(v, X, Z)^α : -(abs(ctg_0(v, X, Z))^α)
    end

    function lrt(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notX = setdiff(collect(1:length(v)), X)
        if isapprox(sum(v[X]), 0., atol=10^-14) || isapprox(sum(v[X]), 1., atol=10^-14) || isapprox(sum(v[notX ∩ Z]), 0., atol=10^-14)
            return missing
        else
            return (sum(v[X ∩ Z]) / sum(v[X])) / (sum(v[notX ∩ Z]) / sum(v[notX]))
        end
    end

    function cheng(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notZ = setdiff(collect(1:length(v)), Z)
        return deltp(v, X, Z) >= 0 ? deltp(v, X, Z) / (1 - (sum(v[X ∩ notZ]) / sum(v[notZ]))) : deltp(v, X, Z) / (sum(v[X ∩ notZ]) / sum(v[notZ]))
    end
    
    function noz(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notX = setdiff(collect(1:length(v)), X)
        (sum(v[X ∩ Z]) / sum(v[X])) - (sum(v[notX ∩ Z]) / sum(v[notX]))
    end
    
    mort(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64}) = sum(v[X ∩ Z]) / sum(v[X]) - sum(v[Z])
    
    function ko(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notX = setdiff(collect(1:length(v)), X)
        (sum(v[X ∩ Z]) / sum(v[X]) - sum(v[notX ∩ Z]) / sum(v[notX])) / (sum(v[X ∩ Z]) / sum(v[X]) + sum(v[notX ∩ Z]) / sum(v[notX]))
    end

    function rips(v::Vector{Float64}, X::Vector{Int64}, Z::Vector{Int64})
        notX = setdiff(collect(1:length(v)), X)
        return dif(v, X, Z) / sum(v[notX])
    end 
end

@everywhere function confSim(ν::Int64)
    @assert ν > 3 # no. of worlds with a minimum of 4
    μ   = rand(1:ν-1) # no. of worlds in H, making sure H is not a tautology
    κ   = rand(1:ν-1) # no. of worlds in E, making sure E is not a tautology
    ϕ   = sample(1:ν, μ, replace=false) # which possible worlds are in H
    ζ   = sample(1:ν, κ, replace=false) # which possible worlds are in E
    τ   = sample(ζ) # pick the actual world, making sure E is veridical
    d   = rand(Dirichlet(ν, 1.))
    b   = issubset(τ, ϕ)
    dc  = dif(d, ϕ, ζ)
    dp  = deltp(d, ϕ, ζ)
    cc  = carn(d, ϕ, ζ)
    cr  = ctg(d, ϕ, ζ, 1.)
    lr  = lrt(d, ϕ, ζ)
    che = cheng(d, ϕ, ζ)
    rp  = rips(d, ϕ, ζ)
    kop = ko(d, ϕ, ζ)
    nz  = noz(d, ϕ, ζ)
    mr  = mort(d, ϕ, ζ)
    fn  = finch(d, ϕ, ζ)
    return b, cc, dp, cr, dc, fn, kop, lr, mr, nz, rp, che
end

@everywhere function lrMod(i::Int64, df::DataFrame)
    dfn   = DataFrame(DV = df[!, 1], IV = df[!, i + 1])
    @rput dfn
    R"""
    m <- suppressWarnings(glm(DV ~ scale(IV), family = binomial(link = "logit"), na.action = na.exclude, data = dfn))
    prob <- predict(m, type=c('response'))
    pred <- prediction(prob, dfn$DV)
    auc <- performance(pred, 'auc')@y.values
    """
    return @rget auc
end

@everywhere function sims(n_worlds::Int64)
    res   = [ confSim(n_worlds) for _ in 1:1000 ]
    df_cf = DataFrame(res)
    return [ lrMod(i, df_cf)[1] for i in 1:size(df_cf, 2) - 1 ]
end

function run_sim(numb_sim::Int64)
    out_ar = Array{Float64,3}(undef, 11, 20, numb_sim)
    for i in 1:numb_sim
        out_ar[:, :, i] = @distributed (hcat) for n in 5:5:100
            sims(n)
        end
    end
    return out_ar
end

out_ar = run_sim(250);

[ pvalue(EqualVarianceTTest(out_ar[3, i, :], out_ar[6, i, :])) for i in 1:20 ]

[ pvalue(EqualVarianceTTest(out_ar[3, i, :], out_ar[11, i, :])) for i in 1:20 ]

function bayes_t_test(data, meas1, meas2, col)
    @rput data
    @rput meas1
    @rput meas2
    @rput col
    R"""
    x <- data[meas1, col, ]
    y <- data[meas2, col, ]
    tt <- suppressMessages(suppressWarnings(extractBF(ttestBF(x = x, y = y), onlybf=TRUE)))
    """
    return @rget tt
end

[ bayes_t_test(out_ar, 3, 6, i) for i in 1:20 ]

[ bayes_t_test(out_ar, 3, 11, i) for i in 1:20 ]

out_auc = mean(out_ar, dims=3);

function aucPlot()
    trace1  = scatter(x=5:5:100, y=out_auc[1, :], mode="lines", name="c")
    trace2  = scatter(x=5:5:100, y=out_auc[2, :], mode="lines", name="s")
    trace3  = scatter(x=5:5:100, y=out_auc[3, :], mode="lines", name="z")
    trace4  = scatter(x=5:5:100, y=out_auc[4, :], mode="lines", name="d")
    trace5  = scatter(x=5:5:100, y=out_auc[5, :], mode="lines", name="r")
    trace6  = scatter(x=5:5:100, y=out_auc[6, :], mode="lines", name="k")
    trace7  = scatter(x=5:5:100, y=out_auc[7, :], mode="lines", name="l")
    trace8  = scatter(x=5:5:100, y=out_auc[8, :], mode="lines", name="m")
    trace9  = scatter(x=5:5:100, y=out_auc[9, :], mode="lines", name="n")
    trace10 = scatter(x=5:5:100, y=out_auc[10, :], mode="lines", name="g")
    layout  = Layout(width=850, height=510, margin=attr(l=70, r=10, t=10, b=70), 
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18)), yaxis=attr(tickfont=attr(size=18)), font_size=20, 
                     annotations=[(x=-0.1, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))])
    data    = [trace1, trace4, trace10, trace6, trace7, trace8, trace9, trace5, trace2, trace3]
    Plot(data, layout)
end
aucPlot()

function aucPlot1()
    trace3  = scatter(x=5:5:100, y=out_auc[3, :], mode="lines", name="z")
    trace6  = scatter(x=5:5:100, y=[], mode="lines", name="k")
    trace11 = scatter(x=5:5:100, y=out_auc[11, :], mode="lines", name="ch")
    layout  = Layout(width=850, height=510, margin=attr(l=70, r=10, t=10, b=70), 
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18)), yaxis=attr(tickfont=attr(size=18)), font_size=20, 
                     annotations=[(x=-0.1, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))])
    data    = [trace11, trace6, trace3]
    Plot(data, layout)
end
aucPlot1()

@everywhere function confSim_weighted(ν::Int64)
    @assert ν > 3 # no. of worlds with a minimum of 4
    μ   = rand(1:ν-1) # no. of worlds in H, making sure H is not a tautology
    κ   = rand(1:ν-1) # no. of worlds in E, making sure E is not a tautology
    d   = rand(Dirichlet(ν, 1.))
    τ   = sample(Weights(d)) # pick the actual world, where more probable worlds have a correspondingly greater chance of being selected
    ζ   = sample(1:ν, κ, replace=false) # which possible worlds are in E
    λ   = unique(push!(ζ, τ)) # make sure E is veridical
    ϵ   = length(λ) == ν ? popfirst!(λ) : λ # make sure E is non-tautological
    ϕ   = sample(1:ν, μ, replace=false) # which possible worlds are in H
    b   = issubset(τ, ϕ)
    dc  = dif(d, ϕ, ζ)
    dp  = deltp(d, ϕ, ζ)
    cc  = carn(d, ϕ, ζ)
    cr  = ctg(d, ϕ, ζ, 1.)
    lr  = lrt(d, ϕ, ζ)
    che = cheng(d, ϕ, ζ)
    rp  = rips(d, ϕ, ζ)
    kop = ko(d, ϕ, ζ)
    nz  = noz(d, ϕ, ζ)
    mr  = mort(d, ϕ, ζ)
    fn  = finch(d, ϕ, ζ)
    return b, cc, dp, cr, dc, fn, kop, lr, mr, nz, rp, che
end

@everywhere function sims_weighted(n_worlds::Int64)
    res   = [ confSim_weighted(n_worlds) for _ in 1:1000 ]
    df_cf = DataFrame(res)
    return [ lrMod(i, df_cf)[1] for i in 1:size(df_cf, 2) - 1 ]
end

function run_sim_weighted(numb_sim::Int64)
    out_ar = Array{Float64,3}(undef, 11, 20, numb_sim)
    for i in 1:numb_sim
        out_ar[:, :, i] = @distributed (hcat) for n in 5:5:100
            sims_weighted(n)
        end
    end
    return out_ar
end

out_ar1 = run_sim_weighted(250);

[ pvalue(EqualVarianceTTest(out_ar1[3, i, :], out_ar1[6, i, :])) for i in 1:20 ]

[ pvalue(EqualVarianceTTest(out_ar1[3, i, :], out_ar1[9, i, :])) for i in 1:20 ]

[ pvalue(EqualVarianceTTest(out_ar1[3, i, :], out_ar1[11, i, :])) for i in 1:20 ]

[ pvalue(EqualVarianceTTest(out_ar1[6, i, :], out_ar1[11, i, :])) for i in 1:20 ]

[ bayes_t_test(out_ar1, 3, 6, i) for i in 1:20 ]

[ bayes_t_test(out_ar1, 3, 11, i) for i in 1:20 ]

[ bayes_t_test(out_ar1, 6, 11, i) for i in 1:20 ]

out_auc0 = mean(out_ar1, dims=3);

function aucPlot2()
    trace1  = scatter(x=5:5:100, y=out_auc0[1, :], mode="lines", name="c")
    trace2  = scatter(x=5:5:100, y=out_auc0[2, :], mode="lines", name="s")
    trace3  = scatter(x=5:5:100, y=out_auc0[3, :], mode="lines", name="z")
    trace4  = scatter(x=5:5:100, y=out_auc0[4, :], mode="lines", name="d")
    trace5  = scatter(x=5:5:100, y=out_auc0[5, :], mode="lines", name="r")
    trace6  = scatter(x=5:5:100, y=out_auc0[6, :], mode="lines", name="k")
    trace7  = scatter(x=5:5:100, y=out_auc0[7, :], mode="lines", name="l")
    trace8  = scatter(x=5:5:100, y=out_auc0[8, :], mode="lines", name="m")
    trace9  = scatter(x=5:5:100, y=out_auc0[9, :], mode="lines", name="n")
    trace10 = scatter(x=5:5:100, y=out_auc0[10, :], mode="lines", name="g")
    layout  = Layout(width=850, height=510, margin=attr(l=70, r=10, t=10, b=70), 
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18)), yaxis=attr(tickfont=attr(size=18)), font_size=20, 
                     annotations=[(x=-0.1, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))])
    data    = [trace1, trace4, trace10, trace6, trace7, trace8, trace9, trace5, trace2, trace3]
    Plot(data, layout)
end
aucPlot2()

function aucPlot3()
    trace3  = scatter(x=5:5:100, y=out_auc0[3, :], mode="lines", name="z")
    trace11 = scatter(x=5:5:100, y=out_auc0[11, :], mode="lines", name="ch")
    trace6  = scatter(x=5:5:100, y=out_auc0[6, :], mode="lines", name="k")
    layout  = Layout(width=850, height=510, margin=attr(l=70, r=10, t=10, b=70), 
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18)), yaxis=attr(tickfont=attr(size=18)), font_size=20, 
                     annotations=[(x=-0.1, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))])
    data    = [trace11, trace6, trace3]
    Plot(data, layout)
end
aucPlot3()

function aucPlot4()
    trace03  = scatter(x=5:5:100, y=out_auc[3, :], mode="lines", name="z")
    trace011 = scatter(x=5:5:100, y=out_auc[11, :], mode="lines", name="ch")
    trace3  = scatter(x=5:5:100, y=out_auc0[3, :], mode="lines", name="z (weighted)")
    trace11 = scatter(x=5:5:100, y=out_auc0[11, :], mode="lines", name="ch (weighted)")
    layout  = Layout(width=950, height=600, margin=attr(l=70, r=10, t=10, b=70), 
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18)), yaxis=attr(tickfont=attr(size=18)), font_size=20, 
                     annotations=[(x=-0.1, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))])
    data    = [trace011, trace11, trace03, trace3]
    Plot(data, layout)
end
aucPlot4()
