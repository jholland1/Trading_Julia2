## Generic Bayesian Neural Networks
#=
The below code is intended for use in more general applications, where you need to be able to change the basic network shape fluidly. The code above is highly rigid, and adapting it for other architectures would be time consuming. Currently the code below only supports networks of `Dense` layers.

    Here, we solve the same problem as above, but with three additional 2x2 `tanh` hidden layers. 
    You can modify the `network_shape` variable to specify differing architectures. 
    A tuple `(3,2, :tanh)` means you want to construct a `Dense` layer with 3 outputs, 2 inputs, 
    and a `tanh` activation function. You can provide any activation function found in Flux by 
    entering it as a `Symbol` (e.g., the `tanh` function is entered in the third part of the tuple as `:tanh`).
=#     
using MarketData, DataFrames, Plots, Statistics, Flux, Optim

# Specify the network architecture.
inputs = 5
nodes = 3
network_shape = 
   [(nodes,inputs, :tanh),
   (nodes,nodes, :tanh), 
   (1,nodes, :tanh)]
    
# Regularization, parameter variance, and total number of
# parameters.
alpha = 0.09
sig = sqrt(1.0 / alpha)

print("Number of Threads: ")
println(Threads.nthreads())

# global curr_loss=0.0f0

start = DateTime(2013, 8, 1)
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
y = convert(Vector{Float32},values(VOO.Open))
# data = [values(VOO.Open) values(VOO.Volume) values(AAPL.Open) values(GLD.Open) values(VIXY.Open)]
# y = [values(VOO.Open)]
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

function rsc(data;n=5)
  r,c=size(data)
  rsc_data=zeros(Float32,(r,c))
  means=zeros(Float32,(r,c))
  stds=zeros(Float32,(r,c))
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
  return rsc_data[1:end-2*n,:],means[1:end-2*n,:],stds[1:end-2*n,:],data[1+n:r-n,1]
end
window = 15
data_rsc,m,s,y_rsc = rsc(data,n=window)
println(typeof(data_rsc))
println(typeof(y_rsc))
plt = scatter(data_rsc, label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY"])
# plt=plot()
lab=["VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"]
for i = 1:size(data)[2]
  plot!(plt,m[:,i], ribbon=2*s[:,i],label=lab[i])
end
# plot!(plt,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY" "VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"])
display(plt)

# layer=RNN
# layer=LSTM
# layer=GRU
# nodes = 1
# inputs=5
# m = Chain(layer(inputs,nodes; init=Flux.glorot_normal),
#                 Dense(nodes,1,tanh;init=Flux.glorot_normal))

#Probably should add warmup iters here
# [m(x[i]) for i in 1:warmup]


num_data = size(data_rsc)[1]
#TODO: THIS IS NOT USING RSC DATA YET...UNSCALED!!!!
seq_len = 10
data_rsc = [data_rsc[i,:] for i in 1:size(data_rsc)[1]]
X = [data_rsc[i-seq_len:i] for i in seq_len+1:num_data-1]
#Vector of sequences, Vector of sequence length, Vector of features 
Y = [[[y_rsc[i]], [y_rsc[i+1]]] for i in seq_len+1:num_data-1]
data_and_target=zip(X,Y)


# evalcb()=@show(curr_loss)
println("Printing type of data_and_target, this should be:")
println("Base.Iterators.Zip{Tuple{Vector{Vector{Vector{Float32}}}, Vector{Vector{Vector{Float32}}}}}")
# data_and_target=(X,Y)
println(typeof(data_and_target))

num_params = sum([i * o + i for (i, o, _) in network_shape])

#compute number of params for rnn networks
num_params_rnn = sum([i * o + i + i * i for (i, o, _) in network_shape])
i,o,_=network_shape[end]
num_params_rnn = num_params_rnn-i*i

# This modification of the unpack function generates a series of vectors
# given a network shape.
function unpack(θ::AbstractVector, network_shape::AbstractVector)
    index = 1
    weights = []
    biases = []
    for layer in network_shape
        rows, cols, _ = layer
        size = rows * cols
        last_index_w = size + index - 1
        last_index_b = last_index_w + rows
        push!(weights, reshape(θ[index:last_index_w], rows, cols))
        push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
        index = last_index_b + 1
    end
    return weights, biases
end

# RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) = 
#   RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))
function unpack_rnn(θ::AbstractVector, network_shape::AbstractVector)
    index = 1
    weights = []
    biases = []
    for layer in network_shape[1:end-1]
        rows, cols, _ = layer

        #same as unpack_rnn but have to push double the weights and biases for the recurrent weights (and biases)
        #Stored as Wi, Wh, bi, bh
        size = rows * cols
        last_index_wi = size + index - 1
        #Push Wi (Input weights)
        push!(weights, reshape(θ[index:last_index_wi], rows, cols))
        size = rows*rows
        last_index_wh = size + last_index_wi - 1
        #Push Wh (Hidden state weights)
        push!(weights, reshape(θ[last_index_wi:last_index_wh], rows, rows))
        last_index_bi = last_index_wh + rows
        #Push bi (Input biases)
        push!(biases, reshape(θ[last_index_wh+1:last_index_bi], rows))
        # last_index_bh = last_index_bi + rows
        #Push hidden state
        push!(biases, zeros(Float32,(rows,1)))
        index = last_index_bi + 1
    end
    #Now do the output layer:
    layer = network_shape[end]
    rows, cols, _ = layer
    size = rows * cols
    last_index_w = size + index - 1
    last_index_b = last_index_w + rows
    push!(weights, reshape(θ[index:last_index_w], rows, cols))
    push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
    index = last_index_b + 1

    return weights, biases
end
    
# Generate an abstract neural network given a shape, 
# and return a prediction.
function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
    weights, biases = unpack(θ, network_shape)
    layers = []
    for i in eachindex(network_shape)
        push!(layers, Dense(weights[i],
            biases[i],
            eval(network_shape[i][3])))
    end
    nn = Chain(layers...)
    return nn(x)
end

function loss(x,y;m=m)
    Flux.reset!(m)
    yp=0.0f0
    for xi in x
      yp=(m(xi)[1]+1.0f0)/2.0f0
      # yp = m(xi)[1]
    end
    curr_loss=1.0f0/softplus(yp*(y[2][1]-y[1][1])/y[1][1]-yp.*(y[1][1]-y[2][1])./y[1][1])
  end
  # Python code for the loss function
  # def loss(model, x, y, training):
  #         y_ = tf.math.tanh(model(x[:-1,:], training=training))
  #         y_ = (tf.transpose(y_)+1.0)/2.0
  #         loss = tf.math.reduce_sum(y_*(y[1:,0]-y[:-1,0])/y[:-1,0]+(1.0-y_)*(y[1:,1]-y[:-1,1])/y[:-1,1])
  #         loss = 1.0/tf.math.softplus(loss)
  #         #loss = 1.0/loss/loss
  #         reg_loss = tf.reduce_sum(model.losses)
  #         if training : loss += reg_loss
  #         return loss

# RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) = 
#   RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))
function build_rnn(θ::AbstractVector, network_shape::AbstractVector)
    weights, biases = unpack_rnn(θ, network_shape)
    layers = []
    for i in eachindex(network_shape[1:end-1])
        push!(layers, RNN(eval(network_shape[i][3]),weights[i*2-1], weights[i*2],biases[i*2-1],biases[i*2]))
    end
    i = length(network_shape)
    push!(layers,Dense(weights[i*2-1], biases[i*2-1],eval(network_shape[i][3])))
    return Chain(layers...)
end

θ=zeros(num_params)
θ_rnn = zeros(Float32,num_params_rnn)

# xs = [1.0f0,2.0f0,3.0f0]
xs = zeros(Float32,inputs)
println("Testing nn_forward()...")
@show nn_forward(xs, θ, network_shape)
println("Testing unpack_rnn")
# unpack_rnn(θ_rnn,network_shape)
println("Testing build_rnn()")
@show rnn = build_rnn(θ_rnn, network_shape)
@show rnn(xs)

function update_weights(θ)
  """Function to update the weights of the RNN"""
  weights, biases = unpack_rnn(θ, network_shape)
  p = params(rnn)
  
  for i in eachindex(network_shape[1:end-1])
      j = i*4-3
      p[j] .= weights[i*2-1] 
      p[j+1] .= weights[i*2]
      p[j+2] .= biases[i*2-1]
      p[j+3] .= biases[i*2]
  end
  i = length(network_shape)
  j = i*4-3
  p[j] .= weights[i*2-1]
  p[j+1] .= biases[i*2-1]
  # println(params(rnn))
end

function l(θ;uw=true)
    """Function to update the weights of the NN input as a vector θ and return the loss
    uw is a boolean that updates the weights if true. Required so the gradient computation
    can compute the gradient without this operation which has (not allowed) mutating array ops"""
    if uw 
      update_weights(θ)
    end
    # Need to have this function return the loss over all of the data
    # Following the example from DiffEqFlux about Bayesian NODEs
    # Details here: https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_NUTS/
    # evalcb() = @show(sum([loss(xi[1],xi[2]) for xi in data_and_target]))
    return -sum([loss(xi[1],xi[2],m=rnn) for xi in data_and_target])-sum(θ.*θ)
end

function lw(p)
  return l(θ_rnn,uw=false)
end

function dldθ(θ)
    """Function to return the gradient of the loss (l) given the current weights"""
    #TODO need to rewrite so that Zygote doesn't throw "mutating arrays not supported error"
    #I think this can be done by creating a function that takes θ and updates the params(rnn), 
    #and then a separate function (that Zygote traces) to compute the loss. Just need to split
    #up the gradient computation from the parameter (mutating array ops) assignments.
    update_weights(θ)
    p = params(rnn)
    x,lambda = Flux.Zygote.pullback(lw,p)
    grad = first(lambda(1w))
    return x, grad
end

println("Testing l(θ)")
# θ_rnn = θ_rnn.+0.9999f0
@show l(θ_rnn)

# function dldθ(θ)
#     #Need function to flatten gradients back into vector (minus the state params one 
#     #per RNN layer, at the 4th index for that layer)
#     x,lambda = Flux.Zygote.pullback(l,θ)
#     grad = first(lambda(1))
    
#     return x, grad
# end

println("Testing dl/dθ")

@show dldθ(θ_rnn)