### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

using Plots, Optim, Flux, DiffEqFlux, StochasticDiffEq,
		DiffEqBase.EnsembleAnalysis, MarketData, DataFrames,
		Statistics,DiffEqSensitivity, Zygote, BenchmarkTools,
		LinearAlgebra

# LinearAlgebra.BLAS.set_num_threads(8)
#plotlyjs()
start = DateTime(2020, 8, 1)
VOO = yahoo(:VOO,YahooOpt(period1=start))
AAPL = yahoo(:AAPL,YahooOpt(period1=start))
GLD = yahoo(:GLD,YahooOpt(period1=start))
VIXY = yahoo(:VIXY,YahooOpt(period1=start))

display(plot(VOO.Open,label="VOO"))

function sc(x)
  y = (values(x).-mean(values(x)))/std(values(x))
end

data = convert(Array{Float32},[sc(VOO.Open) sc(VOO.Volume) sc(AAPL.Open) sc(GLD.Open) sc(VIXY.Open)])
# data = convert(Array{Float32},[sc(VOO.Open) sc(VOO.Volume)] )
# show(data)
# data = Float32[sc(VOO.Open), sc(VOO.Volume)]
display(scatter(timestamp(VOO.Open), data,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY"],xlabel="Date",ylabel="Standardized Value"))

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
window = 5
sde_data,sde_data_vars = avg_and_std(data,n=window)
# show(s)
data=data[window+1:end,:]
# Plots.scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
plt = scatter(data)
for i = 1:size(data)[2]
  plot!(plt,sde_data[:,i], ribbon=2*sde_data_vars[:,i])
end
# plot!(plt,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY" "VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"])
display(plt)

# tspan  = (convert(Float32,0.0), convert(Float32,length(data)))
tspan = (0.0f0,convert(Float32,length(data[:,1])))
tvec = convert(Array{Int32},1:length(data[:,1]))

#tspan

#f = FastChain((x,p) -> x.^3,
#              FastDense(2,50,tanh),
#              FastDense(50,2))


nodes = 50
nInputs = length(data[1,:])
#nInputs = 1
nOutputs = nInputs
init_std=0.001
#nInputs = 2
#nOutputs = 1
# init = Flux.glorot_normal
initer = Flux.sparse_init(sparsity=0.0,std=init_std)
# drift_dudt = FastChain((x,p)-> x, FastDense(nInputs,nodes,tanh,initW = init),FastDense(nodes,nOutputs,initW = init))
# diffusion_dudt = FastChain((x,p)-> x, FastDense(nInputs,nodes,tanh,initW = init),FastDense(nodes,nOutputs,initW = init))
drift_dudt = FastChain((x,p)-> x, FastDense(nInputs,nodes,tanh,initW=initer),FastDense(nodes,nOutputs,initW=initer))
diffusion_dudt = FastChain((x,p)-> x, FastDense(nInputs,nodes,tanh,initW=initer),FastDense(nodes,nOutputs,initW=initer))
# drift_dudt = FastChain((x,p,t;d=data)-> d[min(max(floor(Int32,t),1),length(d[:,1])),:], FastDense(nInputs,nodes,tanh),FastDense(nodes,nOutputs))
# diffusion_dudt = FastChain((x,p,t;d=data)-> d[min(max(floor(Int32,t),1),length(d[:,1])),:], FastDense(nInputs,nodes,tanh),FastDense(nodes,nOutputs))

#drift_dudt(data[1,:])

# u0 = data[1,:]
u0 = convert(Array{Float32},data[1,:])

typeof(u0)

#Was using NeuralDSDE...NeuralSDE doesn't appear to work with any solver settings
#Demo used reltol = abstol = 1e-1...

# prediction0 = neuralsde(u0)


# neuralsde = NeuralDSDE(drift_, diffusion_, tspan,SOSRI(),
#                        saveat = tvec, reltol = 1.0e-1, abstol = 1.0e-1)
					   # sensealg = TrackerAdjoint())
neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI2(),
						saveat=tvec, reltol=1.0e-1,abstol=1.0e-1,
						sensealg=TrackerAdjoint())
prediction0 = neuralsde(u0)

# drift_(u, p, t;d = data) = drift_dudt(d[min(max(floor(Int32,t),1),length(d[:,1])),:], p[1:neuralsde.len])
# diffusion_(u, p, t; d = data) = diffusion_dudt(d[min(max(floor(Int32,t),1),length(d[:,1])),:], p[(neuralsde.len+1):end])
# drift_(u, p, t; d = data) = drift_dudt(d[min(max(floor(Int32,t),1),length(d[:,1])),:], p[1:neuralsde.len])
# diffusion_(u, p, t; d = data) = diffusion_dudt(d[min(max(floor(Int32,t),1),length(d[:,1])),:], p[(neuralsde.len+1):end])
drift_(u,p,t) = drift_dudt(u,p[1:neuralsde.len])
diffusion_(u,p,t) = diffusion_dudt(u,p[(neuralsde.len+1):end])

weight_scale = convert(Float32,0.01)
#NEED TO OPTIMIZE ON SDEPROBLEM, NOT NEURAL DSDE, MAYBE USE GALACTICOPTIM?
prob_neuralsde = SDEProblem(drift_, diffusion_, u0,tspan, neuralsde.p)

ensemble_nprob = EnsembleProblem(prob_neuralsde)

#plot(prediction0)
ensemble_nsol = solve(ensemble_nprob, SOSRI(),EnsembleThreads(), trajectories = 100,
						saveat = tvec)
ensemble_nsum = EnsembleSummary(ensemble_nsol)

plt1 = plot(ensemble_nsum, title = "Neural SDE: Before Training")
display(scatter!(plt1, tvec, data, lw = 3))

θ = neuralsde.p

n = 500
function predict_neuralsde(p, u = u0)
  return Array(neuralsde(u, p))
end

function loss_n_sde(p; n = 100)
  u = repeat(reshape(u0, :, 1), 1, n)
  samples = predict_neuralsde(p, u)
  means = mean(samples, dims = 2)
  vars = var(samples, dims = 2, mean = means)[:, 1, :]
  means = means[:, 1, :]

  loss = sum(abs2, sde_data' - means) + sum(abs2, sde_data_vars' - vars)
  return loss, means, vars
end

l, pred, vars = loss_n_sde(θ)

#Zygote.gradient(l,θ)

iter = 0
cb = function (θ,l,pred;doplot=false)
	global iter

	println("Iterations: ",iter," Loss: ",l)
	iter += 1
#pl = plot(sol)
	return false
end

list_plots = []
iter = 0

# Callback function to observe training
callback = function (p, loss, means, vars; doplot = false)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  # loss against current data
  #display(loss)
  println("Iterations: ",iter," Loss: ",loss)
  # plot current prediction against data
  plt = scatter(tvec, data[:,1],label="VOO")
  scatter!(plt,tvec,data[:,2],label="VOO Vol")
  plot!(plt, tvec, transpose(means), lw = 8, ribbon = transpose(vars), label = "prediction")
  # push!(list_plots, plt)
  display(plt)
  #if doplot
#    display(plt)
 # end
  return false
end


#b(θ,l,pred)

# opt = LBFGS()
opt = ADAM(0.001)

result1 = DiffEqFlux.sciml_train((p) -> loss_n_sde(p,n=n),
                                 neuralsde.p, opt,
                                 cb = callback,
			         allow_f_increases = true,
				 					maxiters = 200)
opt = ADAM(0.001)
#opt = BFGS()
println("Starting Second Optimization")
result2 = DiffEqFlux.sciml_train((p) -> loss_n_sde(p,n=n*2),
							result1.minimizer, opt,cb = callback,
							maxiters = 200)
opt = ADAM(0.001)
println("Starting Third Optimization")
result3 = DiffEqFlux.sciml_train((p) -> loss_n_sde(p,n=n*4),
							result2.minimizer, opt,cb = callback,
							maxiters = 200)

display(result3)

# scatter(timestamp(VOO.Open), data,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY"],xlabel="Date",ylabel="Standardized Value")
# scatter!(timestamp(VOO.Open), result1.u)


samples = [predict_neuralsde(result1.minimizer) for i in 1:100]
means = reshape(mean.([[samples[i][j] for i in 1:length(samples)]
                                      for j in 1:length(samples[1])]),
                    size(samples[1])...)
vars = reshape(var.([[samples[i][j] for i in 1:length(samples)]
                                    for j in 1:length(samples[1])]),
                    size(samples[1])...)

plt2 = scatter(tvec, data,
               label = "data", title = "Neural SDE: After Training",
               xlabel = "Time")
plot!(plt2, tvec, transpose(means), lw = 8, ribbon = transpose(vars), label = "prediction")

plt = plot(plt1, plt2, layout = (2, 1))
display(plt)
savefig(plt, "NN_sde_combined.png"); nothing # sde
