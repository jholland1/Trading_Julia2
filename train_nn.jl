#Test change2.
using MarketData, DataFrames, Plots, Statistics, Flux, Optim
using Flux: @epochs
using Flux: throttle
using Base.Threads

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
l=["VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"]
for i = 1:size(data)[2]
  plot!(plt,m[:,i], ribbon=2*s[:,i],label=l[i])
end
# plot!(plt,label=["VOO" "VOO Volume" "AAPL" "GLD" "VIXY" "VOO Mean" "VOO Vol Mean" "AAPL Mean" "GLD Mean" "VIXY Mean"])
display(plt)

layer=RNN
# layer=LSTM
# layer=GRU
nodes = 1
inputs=5
m = Chain(layer(inputs,nodes; init=Flux.glorot_normal),
                Dense(nodes,1,tanh;init=Flux.glorot_normal))

#Probably should add warmup iters here
# [m(x[i]) for i in 1:warmup]

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
num_data = size(data_rsc)[1]
#TODO: THIS IS NOT USING RSC DATA YET...UNSCALED!!!!
seq_len = 10
data_rsc = [data_rsc[i,:] for i in 1:size(data_rsc)[1]]
X = [data_rsc[i-seq_len:i] for i in seq_len+1:num_data-1]
#Vector of sequences, Vector of sequence length, Vector of features 
Y = [[[y_rsc[i]], [y_rsc[i+1]]] for i in seq_len+1:num_data-1]
data_and_target=zip(X,Y)

evalcb() = @show(sum([loss(xi[1],xi[2]) for xi in data_and_target]))
# evalcb()=@show(curr_loss)
println("Printing type of data_and_target, this should be:")
println("Base.Iterators.Zip{Tuple{Vector{Vector{Vector{Float32}}}, Vector{Vector{Vector{Float32}}}}}")
# data_and_target=(X,Y)
println(typeof(data_and_target))

println("Testing loss function and printing output...")
# println(loss(X[1],Y[1]))
evalcb()
println("Testing Zygote gradient...")
ps = Flux.params(m)
println(gradient(()->loss(X[1],Y[1]),ps))
opt=ADAM(0.001f0)


# NEED THIS TO BE:
# Base.Iterators.Zip{Tuple{Vector{Vector{Vector{Float32}}}, Vector{Vector{Vector{Float32}}}}}
# data_and_target=(X,Y)

function parallel_loss(X,Y;m=m)
  ploss = Threads.Atomic{Float32}(0.0)
  # ploss = 0.0f0
  # sum([loss(xi[1],xi[2]) for xi in data_and_target])
  l = length(X)
  models = [deepcopy(m) for i in 1:Threads.nthreads()]
  Threads.@threads for i in 1:l
    # local_m = deepcopy(m)
    Threads.atomic_add!(ploss,loss(X[i],Y[i],m=models[Threads.threadid()]))
    # ploss+=loss(X[i],Y[i])
  end
  return ploss
end
println("Testing parallel loss...")
parallel_loss(X,Y)
println("Timing serial loss: ")
@time evalcb()
println("Timing parallel loss: ")
@time parallel_loss(X,Y)

## WORK ON THIS!!!
#ps = list of vectors and arrays of parameters
# get gradient by storing grad, then get gradient valuse by grad(ps[1]) for all ps in ps --> grads = [grad[p] for p in ps]
# Can vectorize parameters into a vector by: ps_vec = collect(Iterators.flatten(ps))

#Also can use Flux.destructure
# \theta , re = Flux.destructure(m) gives vector of params in θ
# and then can restructure by re(\theta)

# TODO: Figure out why destructure is giving more elements than
# params, can't use re(θ) if θ has more elements than params, or
# need to find a utility that stores the structure of the params
# so it can be reconstructed
function grad_loss(X,Y;m=m)
  # ploss = Threads.Atomic{Float32}(0.0)
  # ploss = 0.0f0
  # sum([loss(xi[1],xi[2]) for xi in data_and_target])
  l = length(X)
  #These two ways of getting a vector of params (θ) give vecs of
  #different lengths, destructure gives more elements!!! 
  # θ, re = Flux.destructure(m)
  
  θ = collect(Iterators.Flatten(Flux.params(m)))
  
  θ! = [0.0f0 for i in 1:length(θ)]
  println(size(θ!))
  # θ! = [Threads.Atomic{Float32}(0.0) for i in 1:length(θ)]
  temp_mat = [θ! for i in 1:Threads.nthreads()]
  println(size(temp_mat[1]))
  models = [deepcopy(m) for i in 1:Threads.nthreads()]
  # grad = gradient(()->loss(X[1],Y[1],m=models[1]),ps)
  Threads.@threads for i in 1:l
    # local_m = deepcopy(m)
    ps = Flux.params(models[Threads.threadid()])
    # print("ps is: ")
    # println(ps)
    grad = gradient(()->loss(X[i],Y[i],m=models[Threads.threadid()]),ps)
    # print("grad is: ")
    # println(grads)
    grads = [grad[p] for p in ps]
    # print("grads is: ")
    # println(grads)
    grads_vec = collect(Iterators.flatten(grads))
    temp_mat[Threads.threadid()] = grads_vec
    # ploss+=loss(X[i],Y[i])
  end
  # println(size(temp_mat))
  # println(size(temp_mat[1]))
  println(temp_mat)
  for i in 1:length(θ!) 
    θ![i] = sum([temp_mat[j][i] for j in 1:Threads.nthreads()])
  end
  return θ!
end
θ, re = Flux.destructure(m)
function loss_vec(θ; X=X, Y=Y, m=m, re = re) 
  #WORK HERE
  m = re(θ)
  parallel_loss(X,Y,m=m)
end

loss_vec(θ)
println("Timing parallel loss (vector version): ")
@time loss_vec(θ)

println("Testing parallel gradient: ")
grad_loss(X,Y)

println("Timing parallel gradient: ")
@time grad_loss(X,Y)



# println("Calling Flux.train!()...")
# batches = 3
# nbatches = 10
# for i = 1:nbatches
#   print("Batch number ")
#   print(i)
#   print(" of ")
#   println(nbatches)
#   evalcb()
#   @epochs batches Flux.train!(loss,ps,data_and_target,opt)
# end
# evalcb()