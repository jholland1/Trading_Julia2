using Flux
using Flux.Losses: mse
using Flux: @epochs

m = Chain(RNN(2, 5), Dense(5, 1))

# function loss(x, y)
#     m(x[1]) # ignores the output but updates the hidden states
#     sum(mse(m(xi), yi) for (xi, yi) in zip(x[2:end], y))
# end  
# x = rand(Float32, 2)
x = [rand(Float32, 2) for i = 1:3]
# Vector{Vector{Float32}} (alias for Array{Array{Float32, 1}, 1})
# 3-element Vector{Vector{Float32}}:
#  [0.4589591, 0.85384595]
#  [0.31209993, 0.89182806]
#  [0.66939104, 0.4695114]
y = [rand(Float32, 1) for i=1:2]
# Vector{Vector{Float32}} (alias for Array{Array{Float32, 1}, 1})
# 2-element Vector{Vector{Float32}}:
#  [0.82507586]
#  [0.085953]

# println("Testing loss function")
# println(loss(x, y))

#THIS LOSS ONLY TAKES A SINGLE EXAMPLE AT A TIME
function loss(x, y)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
    # print("Loss: ")
    # println(l)
    # return l
end
  
seq_init = [rand(Float32, 2)] #seq_len=1, 2 feat
seq_1 = [rand(Float32, 2) for i = 1:3] #seq_len=3
seq_2 = [rand(Float32, 2) for i = 1:3] ##seq_len=3

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

#loss(X,Y) --> ERROR
#loss(x,y) --> WORKS!!

X = [seq_1, seq_2] # x2 sequences
#Vector of sequences, Vector of sequence length, Vector of features 
Y = [y1, y2]
#Vector of sequences, Vector of sequence length, vector of features
data = zip(X,Y)
# println("Initial Loss: ")
# println(loss(X,Y))
Flux.reset!(m)
[m(x) for x in seq_init]
  
ps = params(m)
opt= ADAM(1e-4)
@epochs 100 Flux.train!(loss, ps, data, opt)
# println("Final Loss: ")
# println(loss(X,Y))