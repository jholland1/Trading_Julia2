#Test change2.
using MarketData, DataFrames, Plots, Statistics, Flux
# using Base.Threads
using Random: seed!
using StatsBase: sample
using JLD
using BSON: @save

include("scaling_funs.jl") #gives us rsc() and sc()
#=
Can use the following outside of VS Code
using ProfileView
@profview profile_test(1)  # run once to trigger compilation (ignore this one)
@profview profile_test(10)
=#

plotlyjs() #use plotlyjs backend for interactive plots

# nepochs = 100
# nsamples = 10
nepochs=500000
nsamples=5000
@assert nepochs > nsamples
# function train_logreg(; model, loss, data, holdout, grad_fun, steps, update)
function train_logreg(;steps, update, samples)
  nodes = 20
  layers = 1
  inputs= 5
  reg_per_weight = 0.0001f0*561f0 #561 corresponds to # of params in a 1L20N network
  # prior_reg = 0.000001f0 #Weight regularization per weight!! 
  dropout = 0.0f0

  nbatches = 1

  layer=Flux.RNNCell
  # layer=Flux.LSTMCell
  # layer=Flux.GRUCell

  # act = NNlib.leakyrelu #Only for layer=Flux.RNNCell
  act = NNlib.tanh_fast

  weight_init=Flux.kaiming_normal(gain=0.0001f0)
  bias_init=Flux.kaiming_normal(gain=0.0001f0)

  #I wouldn't use this...I don't think weight regularization is implemented correctly in SGD and especially SGLD
  reg(x) = prior_reg*sum(xs.^2.0f0 for xs in x) #Regularization function applied to Flux.params(m)

  #seq_len defines the number of timesteps in a batch
  seq_len = 20
  warmup = 5 #defines number of warmup iterations to perform in the batch
  window = 3
  #253 trading days in year
  holdout_batches = 3 #defines number of holdout batches to hold for holdout
  start = DateTime(2013, 8, 1)
  VOO = yahoo(:VOO,YahooOpt(period1=start))
  AAPL = yahoo(:AAPL,YahooOpt(period1=start))
  GLD = yahoo(:GLD,YahooOpt(period1=start))
  VIXY = yahoo(:VIXY,YahooOpt(period1=start))

  #TODO: USING PRESENT DAY VOLUME...THIS IS CHEATING!! (LOOKING FORWARD, NEED TO INDESX BACK)
  data = convert(Array{Float32},[values(VOO.Open[2:end]) values(VOO.Volume[1:end-1]) values(AAPL.Open[2:end]) values(GLD.Open[2:end]) values(VIXY.Open[2:end])])
  y = convert(Vector{Float32},values(VOO.Open[2:end]))

  print("Calling RSC with window=")
  println(window)
  data_rsc,m,s,y_rsc = rsc(data,n=window)
  num_data = size(data_rsc)[1]
  # println(window)
  #TODO: THIS IS NOT USING RSC DATA YET...UNSCALED!!!!

  data_rsc = [data_rsc[i,:] for i in 1:size(data_rsc)[1]]
  #Vector of sequences, Vector of sequence length, Vector of features 
  
  #Throw out warump iters
  # X = [data_rsc[i-seq_len:i] for i in seq_len+1:seq_len:num_data-1]
  # Y = [[y_rsc[i-seq_len+warmup:i], y_rsc[i-seq_len+1+warmup:i+1]] for i in seq_len+1:seq_len:num_data-1]
  #Overlapping batches so no data thrown out in warmup iters
  X = [data_rsc[i-seq_len-warmup:i] for i in seq_len+1+warmup:seq_len:num_data-1]
  Y = [[y_rsc[i-seq_len:i], y_rsc[i-seq_len+1:i+1]] for i in seq_len+1+warmup:seq_len:num_data-1]

  #Overlapping batches so no data thrown out in warmup iters
  batches = length(X)-holdout_batches-1
  if nbatches == 0 || nbatches >= batches
    println("Not using batching (Stochastic Descent)")
    nbatches = batches
  else
    println("Batching using $(nbatches) of $(batches) batches")
  end

  if layers == 1
    if layer == Flux.RNNCell
      m = Chain(Flux.Dropout(dropout),Flux.Recur(layer(inputs,nodes,act,init=weight_init,initb=bias_init)),Flux.Dropout(dropout),Dense(nodes,1,NNlib.tanh_fast))
    else
      m = Chain(Flux.Dropout(dropout),Flux.Recur(layer(inputs,nodes,init=weight_init,initb=bias_init)),Flux.Dropout(dropout),Dense(nodes,1,NNlib.tanh_fast))
    end
  else # layers == 2
    # m = Chain(layer(inputs,nodes),layer(nodes, nodes),Dense(nodes,1,NNlib.tanh_fast))
    if layer == Flux.RNNCell
      m = Chain(Flux.Dropout(dropout),Flux.Recur(layer(inputs,nodes,act,init=weight_init,initb=bias_init)),
        Flux.Dropout(dropout),Flux.Recur(layer(nodes,nodes,act,init=weight_init,initb=bias_init)),Flux.Dropout(dropout),Dense(nodes,1,NNlib.tanh_fast))
    else
      m = Chain(Flux.Dropout(dropout),Flux.Recur(layer(inputs,nodes,init=weight_init,initb=bias_init)),
        Flux.Dropout(dropout),Flux.Recur(layer(nodes,nodes,init=weight_init,initb=bias_init)),Flux.Dropout(dropout),Dense(nodes,1,NNlib.tanh_fast))
    end
  end
  nparams = sum(length,params(m))
  println("Number of parameters in model: $(nparams)")
  prior_reg = reg_per_weight/convert(Float32,nparams)
  
  function loss(x::Vector{Vector{Float32}},y::Vector{Vector{Float32}})
    yp = [(m(xi)[1]+1.0f0)/2.0f0 for xi in x]
    @views curr_loss = sum( @. 1.0f0+yp*(y[2]-y[1])/y[1]+(1.0f0-yp)*(y[1]-y[2])/y[1]) #percent return
    curr_loss /= convert(Float32, length(x)) #average percent return
    curr_loss = 1.0f0/curr_loss #Inverse (minimize inverse of average percent return)
    return curr_loss::Float32
  end

  function return_estimate(x::Vector{Vector{Float32}},y::Vector{Vector{Float32}})
    yp = [round((m(xi)[1]+1.0f0)/2.0f0) for xi in x]
    @views curr_return = sum( @. yp*(y[2]-y[1])/y[1]+(1.0f0-yp)*(y[1]-y[2])/y[1]) #return
    curr_return /= convert(Float32, length(x)) #average daily return for batch
    return curr_return::Float32
  end

  function loss_serial(X,Y)
    """
    Computes the loss function without any regularization.
      Zygote has issues differentiating both the RNN output and analysis
      prior loss on the weights of the NN. Therefore, regularization portion
      of the loss is computed outside of Zygote. This is also convenient
      for batching, where the gradient of the batch is multiplied by the 
      batch ratio, but the regularization loss should not be.
    """
    loss_total = 0.0f0
    Flux.reset!(m)
    # regularizer = sum(reg,Flux.params(m))
    for i in 1:length(X)
      # Flux.reset!(m) 
      [m(X[i][w]) for w in 1:warmup]
      loss_total+=loss(X[i][warmup+1:end],Y[i])
    end
    # Flux.reset!(m)
    # println(loss_total)
    return loss_total/convert(Float32,length(X))#+regularizer
  end

  function annual_return(X,Y)
    """
    Same as loss_serial but returns annual return estimate
    """
    return_total = 0.0f0
    for i = 1:length(X)
      Flux.reset!(m) 
      [m(X[i][w]) for w in 1:warmup]
      return_total+=return_estimate(X[i][warmup+1:end],Y[i])
    end
    # avg_daily_return=return_tota
    ((return_total/convert(Float32,length(X))+1.0f0)^253f0-1.0f0)*100f0 
  end
  # SGLD optimization, copied from https://sebastiancallh.github.io/post/langevin/

    """
    Train the network
    θ = Trainable parameters of the model in vector from
    model = The Flux model
    loss = Function that takes data[1] = X and data[2] = Y and returns the current loss
    data = [X,Y], where X is the inputs to the model and Y are the outputs
    holdout = [X_holdout,Y_holdout] holdout data
    grad_fun = function that takes the data and returns the gradient of the loss w.r.t. θ
    steps = number of training steps to takes
    update = update function defining what optimization method we are using (only SGLD implemented currently)
    """
    seed!(1)

    # paramvec(θ) = reduce(hcat, θ)
    paramvec(θ) = transpose(collect(Iterators.flatten(θ)))
    # model = Dense(length(features), 1, sigmoid) #|> gpu
    Flux.reset!(m)
    θ = Flux.params(m)
    θ₀ = paramvec(θ)
    # predict(x; thres = .5) = model(x) .> thres
    # accuracy(x, y) = mean(cpu(predict(x)) .== cpu(y))

    # loss(x, y) = mean(Flux.binarycrossentropy.(model(x), y))
    train_X = X[1:end-holdout_batches-1]
    train_Y = Y[1:end-holdout_batches-1]
    test_X = X[end-holdout_batches:end]
    test_Y = Y[end-holdout_batches:end]
    trainloss() = loss_serial(train_X, train_Y)
    testloss() = loss_serial(test_X, test_Y)
    # println(testreturn)
    testreturn() = annual_return(test_X, test_Y)
    s = sort(sample(1:batches,nbatches,replace=false))
    ∇L = gradient(()->loss_serial(train_X[s],train_Y[s]), θ)
    
    trainlosses = [trainloss()+sum(reg,θ₀); zeros(steps)]
    # testlosses = [testloss(); zeros(steps)]
    testreturns = [testreturn(); zeros(steps)]
    weights = [θ₀; zeros(nsamples, length(θ₀))]
    batch_ratio = convert(Float32,nbatches)/convert(Float32,batches)

    for t in 1:steps
      # ∇L denotes gradient with respect to loss
      s = sort(sample(1:batches,nbatches,replace=false))
      Flux.trainmode!(m)
      # Flux.reset!(m)
      θ = Flux.params(m)
      ∇L = gradient(()->loss_serial(train_X[s],train_Y[s]), θ)
      # foreach(θᵢ -> update(∇L, θᵢ, t), θ)
      # ∇L = grad_loss(X,Y)
      # Flux.reset!(m)
      # θ = Flux.params(m)
      # println(θ)
      foreach(θᵢ -> update(∇L, θᵢ, t, batch_ratio, prior_reg), θ)    
      # Bookkeeping
      if t >= steps-nsamples
        weights[t-(steps-nsamples)+1, :] = paramvec(θ)
      end
      
      # shift!(weights)
      Flux.testmode!(m)
      if prior_reg>0.0f0
        regularizer = sum(reg,θ)
      else 
        regularizer = 0.0f0
      end
      # println("Regularization Loss is: $(regularizer)")
      trainlosses[t+1] = trainloss() + regularizer
      testreturns[t+1] = testreturn()
      @assert isnan(trainlosses[t+1])==false #terminate if something has gone terribly wrong
      if mod(t, 10) == 0
        println("Iteration: $(t) Loss: $(trainlosses[t+1]) Reg Loss: $(regularizer) Holdout Annual Return: $(testreturns[t+1])%")
      end 
    end
    ## Do some analysis
    # println("Final parameters are $(θ))")
    # println("Test accuracy is $(accuracy(test_X, test_y))")
    println("Average annual return of final $(samples) samples was $(mean(testreturns[end-samples:end]))")
    println("Standard deviation of annual return of final $(samples) samples was $(std(testreturns[end-samples:end]))")
    # println("Testing ensemble model...")

    m, weights, trainlosses, testreturns, window #For some reason window disappears from global scope if not returned and saved here...bug in Julia REPL??
    
end

sgd(∇L, θᵢ, t, br, pr, η = 0.001) = begin
  Δθᵢ = η*(br*∇L[θᵢ] .+ pr.*2.0f0.*θᵢ) #Second term is gradient due to prior loss on weights
  θᵢ .-= Δθᵢ 
  return θᵢ
end
#default a=10, b=1000, γ=0.9
sgld(∇L, θᵢ, t, br, pr, a = 10.0f0, b = 10000.0f0, γ = 0.9f0) = begin
  ϵ = a*(b + t)^-γ
  η = ϵ.*randn(Float32,size(θᵢ))
  Δθᵢ = ϵ.*pr.*θᵢ .+ br*0.5f0ϵ*∇L[θᵢ] + η #Prior loss gradient+gradient term+randomness
  θᵢ .-= Δθᵢ
end

results = train_logreg(steps = nepochs, update = sgld, samples = nsamples)
model, weights, trainlosses, testreturns, window = results;

@save "model.bson" model #JLD format didn't seem to work for the model itself...using BSON
save("training_results.jld","weights",weights, "trainlosses",trainlosses,"testreturns",testreturns,"window",window)

plot(trainlosses)
xlabel!("Epoch")
display(ylabel!("Training Losss"))

histogram(trainlosses[end-nsamples:end])
display(xlabel!("Sampled Losses (Training Set)"))

plot(testreturns)
xlabel!("Epoch")
display(ylabel!("% Annual Return (Holdout Data)"))

histogram(testreturns[end-nsamples:end])
display(xlabel!("Sampled Annual Return (Holdout Set)"))

include("backtest.jl")