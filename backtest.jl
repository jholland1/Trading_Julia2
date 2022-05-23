using Plots, JLD, Flux, CSV, DataFrames, Dates, TimeZones, BusinessDays, Random
using BSON: @load

include("scaling_funs.jl") #Gets us rsc() and sc()

#TODO I thinkj the best way to do this is to just iterate from bt_start->bt_end by a number of minutes (multiple of 45, e.g. 15)
# check each time if A. Day = Trading Day and B. Time = Trading time (maybe a list of times of day). If so, then gather the data 
# at the nearest index in the past. We should probably only do this once for efficiency and save the data for later weight samples
# We will need to scale everything as well. I think we will have to do this for each vector because comparisons/slicing based
# on time is not working very well. This is going to suck...might want to do this in Python honestly..

bias = [0.0f0, 0.5f0, 1.0f0, 2.0f0] #Number of standard deviations to bias towards long position 
warmup = 5
bt_end = round(now(tz"America/New_York"),Minute(15))
bt_start = bt_end - Day(365)

tr = load("training_results.jld")
@load "model.bson" model

# api_tz = "UTC"
# backtest_tz = "America/New_York"

#Using Plots because Makie.jl does not support plotting DateTime as x-axis (yet)
# Use PlotlyJS backend
plotlyjs()

df=Dates.DateFormat("yyyy-mm-dd HH:MM:SSzzzz")
voo=CSV.read("Data/VOO_min.csv",DataFrame,dateformat=df)
sh=CSV.read("Data/SH_min.csv",DataFrame,dateformat=df)
gld=CSV.read("Data/GLD_min.csv",DataFrame,dateformat=df)
aapl=CSV.read("Data/AAPL_min.csv",DataFrame,dateformat=df)
vixy=CSV.read("Data/VIXY_min.csv",DataFrame,dateformat=df)

voo_day=CSV.read("Data/VOO_day.csv",DataFrame,dateformat=df)

data = [voo, sh, aapl, gld, vixy]

#List of times that trading decisions or orders will be made
# trading_times = [Time(9,45)]
trading_times = [Time("09-45-AM","HH-MM-p")]
cal = BusinessDays.USNYSE()
BusinessDays.initcache(cal) #Not necessary, but supposedly speeds things up

#Convert to timezone aware dataframe
#ASSUMES SAVED IN UTC, adds UTC timezone then converts to Eastern (should capture DST correctly...I think)
function add_tz!(d::DataFrame)
    transform!(d, :timestamp => ByRow(x -> astimezone(ZonedDateTime(x,tz"UTC"),tz"America/New_York")) => :timestamp_tz)
    transform!(d, :timestamp_tz => ByRow(x -> DateTime(x)) => :timestamp)
end
# transform!(voo, :timestamp => ByRow(x -> astimezone(ZonedDateTime(x,tz"UTC"),tz"America/New_York")) => :timestamp_tz)
# transform!(voo, :timestamp_tz => ByRow(x -> DateTime(x)) => :timestamp)
# add_tz!(voo)
# add_tz!(sh)
# add_tz!(gld)
# add_tz!(aapl)
# add_tz!(vixy)
data = [add_tz!(dd) for dd in data]
add_tz!(voo_day)

function parse_trading_data(D::DataFrame, trading_times::Vector{Time};cal = cal,lag_bars=0,slip_bars=0,
    bt_start=bt_start,bt_end=bt_end)
    """Function to return data required from array at the specified time. Should 
    iterate in time from bt_start to bt_end and return the data (from just before
    the trading times) and the prices of equities to be traded at just after the 
    trading times (to simulate slippage)

    VERY simple slippage model:
    lag_bars = # bars in past that we make trading decisions on
    slip_bars = # of bars in the future actually executes the traded"""
    Dd = DataFrame() #Data that we make the trading decision on 
    Dt = DataFrame() #Data that we make the trade on
    nbars = size(D,1)
    cur_day=Date(D.timestamp_tz[1])
    ti = 1
    for ii = 1:nbars
        if cur_day != Date(D.timestamp_tz[ii])
            cur_day = Date(D.timestamp_tz[ii])
            ti = 1
            # println("It's a new day!!")
        end
        if isbday(:USNYSE,Date(D.timestamp_tz[ii])) && ti <= length(trading_times) &&
            cur_day < Date(bt_end) && cur_day > Date(bt_start) 
            if Time(D.timestamp_tz[ii]) >= trading_times[ti] && ii >= lag_bars+1 && 
                ii+slip_bars<=nbars #&& abs(Time(D.timestamp[ii])-trading_times[ti]) <= Minute(1)
                # We found a trading time!
                # println(ii)
                append!(Dd,DataFrame(D[ii-lag_bars,:]))
                append!(Dt,DataFrame(D[ii+slip_bars,:]))
                ti+=1
            end
        end
    end
    return Dd, Dt
end

# Even with 5min data missing a lot of bars (not getting a bar every 5 min)...
# might need to subscribe to the data api_tz
lb = 1
sb = 1
# voo_d, voo_t = parse_trading_data(voo,trading_times,cal=cal,lag_bars=lb,slip_bars=sb)
# sh_d, sh_t = parse_trading_data(sh,trading_times,cal=cal,lag_bars=lb,slip_bars=sb)
# gld_d, gld_t = parse_trading_data(gld,trading_times,cal=cal,lag_bars=lb,slip_bars=sb)
# aapl_d, aapl_t = parse_trading_data(aapl,trading_times,cal=cal,lag_bars=lb,slip_bars=sb)
# vixy_d, vixy_t = parse_trading_data(vixy,trading_times,cal=cal,lag_bars=lb,slip_bars=sb)
pdata = [parse_trading_data(dd,trading_times,cal=cal,lag_bars=lb,slip_bars=sb) for dd in data]

#Get rid of day values before pdata values:
#Same index in voo_day is one behind pdata (looking at previous day's bar for volume)
dindices=[Date(d.timestamp) >= Date(pdata[1][1].timestamp[1]) &&  
    Date(d.timestamp) < Date(pdata[1][1].timestamp[end]) for d in eachrow(voo_day)]
dindices[findfirst(dindices)-1]=true
voo_day = voo_day[dindices,:]

# Uncommenting this should free up some memory
voo = nothing
gld = nothing
aapl = nothing
vixy = nothing
sh = nothing

# scatter(pdata[1][1].timestamp,pdata[1][1].open,label="Data")
# scatter!(pdata[1][2].timestamp,pdata[1][2].open,label="Trades",show=true)
tr["weights"] = convert(Matrix{Float32},tr["weights"])
nsamples, nweights = size(tr["weights"])
best_loss, ibest = findmin(tr["trainlosses"])
println("Best loss: $(best_loss)")
println("Test return for best loss sample: $(tr["testreturns"][ibest])")

best_test, ibesttest = findmax(tr["testreturns"])
println("Best holdout data returns: $(best_test)")
println("Training loss for best test sample: $(tr["trainlosses"][ibesttest])")

#Write a function here that iterates through the data and returns a vector of the buy/sell signals each day


Flux.reset!(model)
# θ, re = Flux.destructure(model)

#data = convert(Array{Float32},[values(VOO.Open) values(VOO.Volume) values(AAPL.Open) values(GLD.Open) values(VIXY.Open)])
# data = [voo, sh, aapl, gld, vixy]

idata=convert(Array{Float32},[values(pdata[1][1].close) values(voo_day.volume) values(pdata[3][1].close) values(pdata[4][1].close) values(pdata[5][1].close)])
# y = convert(Vector{Float32},values(VOO.Open))

# print("Calling RSC with window=")
# println(window)
# data_rsc,m,s,y_rsc = rsc(data,n=window)
data_rsc,m,s,y_rsc = rsc(idata,n=tr["window"])

nsamples = size(tr["weights"])[1]
nobs = size(data_rsc)[1]
signals = zeros(Float32, nsamples,nobs)

Flux.reset!(model)
Flux.testmode!(model)
θ, re = Flux.destructure(model)

function get_decisions(mi,data_rsc)
    Flux.reset!(mi)
    Flux.testmode!(mi)
    [mi(convert(Vector{Float32},row))[1] for row in eachrow(data_rsc)]
end

#Go get each weight samples' decisions, can do multi-threaded with a deepcopy
#Results stored in matrix with each row being a samples' decisions
Threads.@threads for i = 1:size(tr["weights"])[1]
    signals[i,:]=get_decisions(deepcopy(re(tr["weights"][i,:])),data_rsc)
end

signal = [mean(s) for s in eachcol(signals)]
signal_std = [std(s) for s in eachcol(signals)]


#Trading starts on day 2*window for the scaler (rsc removes 2*n values from rsc_data)
longe = pdata[1][2][2*tr["window"]+1:end,:]
shorte = pdata[2][2][2*tr["window"]+1:end,:]

stocks = [longe, shorte]

#TODO: Need to disregard the warmup iterations
function run_backtest(b = 0.0f0)
    pvalue = zeros(Float32,length(signal))
    pvalue[1] = 10000f0

    for i = 1:length(signal)-1
        if signal[i] >= 0.0f0 + b*signal_std[i]
            istock = 1
        else 
            istock = 2
        end
        nshares = fld(pvalue[i],stocks[istock].close[i])
        if i > warmup 
            pvalue[i+1] = pvalue[i]+nshares*(stocks[istock].open[i+1]-stocks[istock].close[i])
        else 
            pvalue[i+1] = pvalue[i]
        end
    end
    return pvalue
end
p1 = plot(stocks[1].timestamp[warmup:end],stocks[1].close[warmup:end]./stocks[1].open[warmup].*100.0f0.-100.0f0,label="benchmark",lw=3.0)
for b in bias
    pvalue = run_backtest(b)
    plot!(stocks[1].timestamp[warmup:end],pvalue[warmup:end]./pvalue[warmup].*100.0f0.-100.0f0,label="portfolio, bias=$(b)")
end
title!("Performance")
ylabel!("% Change")

# p2 = scatter(stocks[1].timestamp,signals)
# p2 = plot(stocks[1].timestamp,signal,yerror=signal_std,label="Mean")
p2 = plot(stocks[1].timestamp[warmup:end],signal[warmup:end],ribbon=signal_std,fillalpha=0.5,label="signal")
title!("Decisions")
ylabel!("Signal")
plot(p1, p2, layout=(2,1))