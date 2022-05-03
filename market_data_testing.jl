using MarketData, DataFrames, Plots, Statistics

start = DateTime(2020, 8, 1)
VOO = yahoo(:VOO,YahooOpt(period1=start))
AAPL = yahoo(:AAPL,YahooOpt(period1=start))
GLD = yahoo(:GLD,YahooOpt(period1=start))
VIXY = yahoo(:VIXY,YahooOpt(period1=start))

display(plot(VOO.Open,label="VOO"))

function sc(x)
  y = (values(x).-mean(values(x)))/std(values(x))
end

# data = convert(Array{Float32},[sc(VOO.Open) sc(VOO.Volume) sc(AAPL.Open) sc(GLD.Open) sc(VIXY.Open)])
data = convert(Array{Float32},[values(VOO.Open) values(VOO.Volume) values(AAPL.Open) values(GLD.Open) values(VIXY.Open)])
# data = convert(Array{Float32},[sc(VOO.Open) sc(VOO.Volume)] )
# show(data)
# data = Float32[sc(VOO.Open), sc(VOO.Volume)]
display(scatter(timestamp(VOO.Open), data,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY"],xlabel="Date",ylabel="Standardized Value"))

# display(data)

# moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
function avg_and_std(data;n=5)
  r,c=size(data)
  means=zeros(r-n,c)
  stds=zeros(r-n,c)
  for j = 1:c
    for i = 1:r-n
      means[i,j] = mean(data[i:i+n,j])
      stds[i,j] = std(data[i:i+n,j])
    end
  end
  return means, stds
end

function rsc(data;n=5)
  r,c=size(data)
  rsc_data=zeros(r,c)
  means=zeros(r,c)
  stds=zeros(r,c)
  for j = 1:c
    for i = 1:r-n
      rsc_data[i,j] = (data[i+n,j]-mean(data[i:i+n,j]))/std(data[i:i+n,j])
      # stds[i,j] = std(rsc_data[i:i+n,j])
    end
  end
  for j = 1:c
    for i = 1:r-2*n
      means[i,j] = mean(rsc_data[i:i+n,j])
      stds[i,j] = std(rsc_data[i:i+n,j])
    end
  end
  return rsc_data[1:end-2*n,:],means[1:end-2*n,:],stds[1:end-2*n,:]
end
window = 15
data_rsc,m,s = rsc(data,n=window)
# show(s)
# data=data[window+1:end,:]
# Plots.scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
# plt = scatter(data)
# plt = scatter(m)
plt = scatter(data_rsc)
# plt=plot()
for i = 1:size(data)[2]
  plot!(plt,m[:,i], ribbon=2*s[:,i])
end
# plot!(plt,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY" "VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"])
display(plt)
