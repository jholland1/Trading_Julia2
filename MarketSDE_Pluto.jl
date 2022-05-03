### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ f2b6eb06-1bd9-11eb-3fc3-c918c99cc278
using Plots, Optim, Flux, DiffEqFlux, OrdinaryDiffEq,DifferentialEquations, LaTeXStrings, GalacticOptim,BlackBoxOptim, DiffEqSensitivity, Zygote, MarketData, DataFrames, Statistics, DiffEqBase.EnsembleAnalysis

# ╔═╡ 1adb4546-1bda-11eb-2d36-a1675a64a328
begin
	start = DateTime(2018, 1, 1)
	VOO = yahoo(:VOO,YahooOpt(period1=start))
	AAPL = yahoo(:AAPL,YahooOpt(period1=start))
	GLD = yahoo(:GLD,YahooOpt(period1=start))
	VIXY = yahoo(:VIXY,YahooOpt(period1=start))
end

# ╔═╡ 25568af8-1bda-11eb-2989-4fe1a303144e
plot(VOO.Open,label="VOO")

# ╔═╡ 2915efda-1bda-11eb-0a09-d13b4ff7568d
function sc(x)
  y = (values(x).-mean(values(x)))/std(values(x))
end

# ╔═╡ 2e3e0fbc-1bda-11eb-3afa-2f5f1009319c
begin
	data = convert(Array{Float32},[sc(VOO.Open) sc(VOO.Volume) sc(AAPL.Open) sc(GLD.Open) sc(VIXY.Open)])
	scatter(timestamp(VOO.Open), data,label=["VOO" "VOO Volume" "AAPL" "GLD" 				"VIXY"],xlabel="Date",ylabel="Standardized Value")
end

# ╔═╡ 40b90ab4-1bda-11eb-1a2d-d3b3fab14025
begin
	tspan  = (convert(Float32,0.0), convert(Float32,length(data)))
	tvec = 0:length(data[:,1])-1
end

# ╔═╡ acbc1b96-1be4-11eb-2a8f-99cf00925cfc
tspan

# ╔═╡ a7331be6-1be3-11eb-2fce-596ceb933705
f = FastChain((x,p) -> x.^3,
              FastDense(2,50,tanh),
              FastDense(50,2))

# ╔═╡ 4a0c9f2c-1bda-11eb-02d5-75b2da4ba534
begin
	nodes = 20
	nInputs = 5
	nOutputs = nInputs
	drift_dudt = FastChain(FastDense(nInputs,nodes,tanh),FastDense(nodes,nOutputs))
	diffusion_dudt = FastChain(FastDense(nInputs,nodes,tanh),FastDense(nodes,nOutputs))
#	drift_dudt(data[1,:])
end

# ╔═╡ 2c52670a-1bda-11eb-263a-45a0b5afd356
u0 = convert(Array{Float32},data[1,:])

# ╔═╡ 9bab0b7e-1be0-11eb-24c1-9f26e6e34346
typeof(u0)

# ╔═╡ aaa9a6e0-1bda-11eb-3083-7f9d18681cdd
neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI(),
                       saveat = tvec, reltol = 1e-1, abstol = 1e-1)

# ╔═╡ af1cfc40-1bda-11eb-1504-b1901ab78609
prediction0 = neuralsde(u0)

# ╔═╡ 83317f16-1bda-11eb-11c9-cf4d17e48e8f
begin
	drift_(u, p, t) = drift_dudt(u, p[1:neuralsde.len])
	diffusion_(u, p, t) = diffusion_dudt(u, p[(neuralsde.len+1):end])
end

# ╔═╡ 89ba71e4-1bda-11eb-100e-21f6cf3b9b00
prob_neuralsde = SDEProblem(drift_, diffusion_, u0,(0.0f0, 1.2f0), neuralsde.p)

# ╔═╡ 8f624c02-1bda-11eb-1028-591b07f2bd04
plot(prediction0)

# ╔═╡ c520933a-1bda-11eb-1d3a-7d678450f6e4
θ = neuralsde.p

# ╔═╡ ccb7d950-1bda-11eb-1f42-8506daea3d23
function predict_neuralsde(p)
  return Array(neuralsde(u0, p))
end

# ╔═╡ d1484c98-1bda-11eb-06ab-717736055f98
predict_neuralsde(θ)

# ╔═╡ d4c99912-1bda-11eb-29ff-3f66ae25c7c8
function loss_n_sde(θ)
    pred = predict_neuralsde(θ)
    loss = sum(abs2,data[:,1] .- pred[1,:])
    #loss = sum(abs2,pred[1,:])
    loss,pred
end

# ╔═╡ d8f49b0e-1bda-11eb-3ca1-b1602e165e76
l, pred = loss_n_sde(θ)

# ╔═╡ 056b9108-1be5-11eb-350a-9ba0f22c9708
Zygote.gradient(l,θ)

# ╔═╡ dceb2dd6-1bda-11eb-2989-717524d25ae5


# ╔═╡ 92c8d630-1bdb-11eb-2132-0fbb039954f5
cb = function (θ,l,pred;doplot=false)
	display(l)
	#pl = plot(sol)
	return false
end

# ╔═╡ a0dbf0a4-1bdb-11eb-1eae-8558d51619b4
cb(θ,l,pred)

# ╔═╡ db3f5158-1bdd-11eb-298c-cb55eae80120
opt = ADAM(0.025)

# ╔═╡ e5eb0744-1bda-11eb-3923-0d55decbe843
result1 = DiffEqFlux.sciml_train((p) -> loss_n_sde(p),
                                 neuralsde.p, opt,
                                 cb = cb, maxiters = 1)

# ╔═╡ e54e2e78-1bdd-11eb-26c6-35dc1e754647
begin
	scatter(timestamp(VOO.Open), data,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY"],xlabel="Date",ylabel="Standardized Value")
	scatter!(timestamp(VOO.Open), result1.u)
end

# ╔═╡ Cell order:
# ╠═f2b6eb06-1bd9-11eb-3fc3-c918c99cc278
# ╠═1adb4546-1bda-11eb-2d36-a1675a64a328
# ╠═25568af8-1bda-11eb-2989-4fe1a303144e
# ╠═2915efda-1bda-11eb-0a09-d13b4ff7568d
# ╠═2e3e0fbc-1bda-11eb-3afa-2f5f1009319c
# ╠═40b90ab4-1bda-11eb-1a2d-d3b3fab14025
# ╠═acbc1b96-1be4-11eb-2a8f-99cf00925cfc
# ╠═a7331be6-1be3-11eb-2fce-596ceb933705
# ╠═4a0c9f2c-1bda-11eb-02d5-75b2da4ba534
# ╠═2c52670a-1bda-11eb-263a-45a0b5afd356
# ╠═9bab0b7e-1be0-11eb-24c1-9f26e6e34346
# ╠═aaa9a6e0-1bda-11eb-3083-7f9d18681cdd
# ╠═af1cfc40-1bda-11eb-1504-b1901ab78609
# ╠═83317f16-1bda-11eb-11c9-cf4d17e48e8f
# ╠═89ba71e4-1bda-11eb-100e-21f6cf3b9b00
# ╠═8f624c02-1bda-11eb-1028-591b07f2bd04
# ╠═c520933a-1bda-11eb-1d3a-7d678450f6e4
# ╠═ccb7d950-1bda-11eb-1f42-8506daea3d23
# ╠═d1484c98-1bda-11eb-06ab-717736055f98
# ╠═d4c99912-1bda-11eb-29ff-3f66ae25c7c8
# ╠═d8f49b0e-1bda-11eb-3ca1-b1602e165e76
# ╠═056b9108-1be5-11eb-350a-9ba0f22c9708
# ╟─dceb2dd6-1bda-11eb-2989-717524d25ae5
# ╠═92c8d630-1bdb-11eb-2132-0fbb039954f5
# ╠═a0dbf0a4-1bdb-11eb-1eae-8558d51619b4
# ╠═db3f5158-1bdd-11eb-298c-cb55eae80120
# ╠═e5eb0744-1bda-11eb-3923-0d55decbe843
# ╠═e54e2e78-1bdd-11eb-26c6-35dc1e754647
