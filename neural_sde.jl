using Plots, Statistics
using DiffEqFlux, StochasticDiffEq, Plots, DiffEqBase.EnsembleAnalysis

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.0f0)
#This is new...
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueSDEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
#This was deleted...
#t = range(tspan[1],tspan[2],length=datasize)
mp = Float32[0.2,0.2]
function true_noise_func(du,u,p,t)
    du .= mp.*u
end
prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)

# Take a typical sample from the mean
#Removed MonteCarloProblem from this example
#monte_prob = MonteCarloProblem(prob)
#monte_sol = solve(monte_prob,SOSRI(),num_monte = 100)
#monte_sum = MonteCarloSummary(monte_sol)
#sde_data = Array(timeseries_point_mean(monte_sol,t))

# Take a typical sample from the mean
#This whole implementation is new
ensemble_prob = EnsembleProblem(prob_truesde)
ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10000)
ensemble_sum = EnsembleSummary(ensemble_sol)

sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))
#Looks slightly different all the way down so just deleting and replacing everything

drift_dudt = FastChain((x, p) -> x.^3, FastDense(2, 50, tanh),
                       FastDense(50, 2))
diffusion_dudt = FastChain(FastDense(2, 2))

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI(),
                       saveat = tsteps, reltol = 1e-1, abstol = 1e-1)
                       # Get the prediction using the correct initial condition
prediction0 = neuralsde(u0)

drift_(u, p, t) = drift_dudt(u, p[1:neuralsde.len])
diffusion_(u, p, t) = diffusion_dudt(u, p[(neuralsde.len+1):end])

prob_neuralsde = SDEProblem(drift_, diffusion_, u0,(0.0f0, 1.2f0), neuralsde.p)

ensemble_nprob = EnsembleProblem(prob_neuralsde)
ensemble_nsol = solve(ensemble_nprob, SOSRI(), trajectories = 100,saveat = tsteps)
ensemble_nsum = EnsembleSummary(ensemble_nsol)

plt1 = plot(ensemble_nsum, title = "Neural SDE: Before Training")
scatter!(plt1, tsteps, sde_data', lw = 3)

scatter(tsteps, sde_data[1,:], label = "data")
scatter!(tsteps, prediction0[1,:], label = "prediction")

function predict_neuralsde(p, u = u0)
  return Array(neuralsde(u, p))
end

function loss_neuralsde(p; n = 100)
  u = repeat(reshape(u0, :, 1), 1, n)
  samples = predict_neuralsde(p, u)
  means = mean(samples, dims = 2)
  vars = var(samples, dims = 2, mean = means)[:, 1, :]
  means = means[:, 1, :]
  loss = sum(abs2, sde_data - means) + sum(abs2, sde_data_vars - vars)
  return loss, means, vars
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
  display(loss)

  # plot current prediction against data
  plt = Plots.scatter(tsteps, sde_data[1,:], yerror = sde_data_vars[1,:],
                     ylim = (-4.0, 8.0), label = "data")
  Plots.scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
  push!(list_plots, plt)

  if doplot
    display(plt)
  end
  return false
end

opt = ADAM(0.025)

# First round of training with n = 10
result1 = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 100),neuralsde.p, opt,cb = callback, maxiters = 100)
result2 = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 1000),result1.u, opt,cb = callback, maxiters = 100)

_, means, vars = loss_neuralsde(result2.u, n = 1000)

plt2 = Plots.scatter(tsteps, sde_data', yerror = sde_data_vars',
                     label = "data", title = "Neural SDE: After Training",
                     xlabel = "Time")
plot!(plt2, tsteps, means', lw = 8, ribbon = vars', label = "prediction")

plt = plot(plt1, plt2, layout = (2, 1))
savefig(plt, "NN_sde_combined.png")
display(plt); nothing
#; nothing # sde
