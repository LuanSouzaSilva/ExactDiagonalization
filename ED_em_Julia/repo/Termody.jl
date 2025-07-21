using Pkg
using Combinatorics
using LinearAlgebra
using SparseArrays
using Arpack
using Plots
using BenchmarkTools

using CUDA

include("ExactDiag.jl")

function Partition(energy_levels, beta)
    return sum(exp.(-energy_levels.*beta))
end

function Mean_Energy(energy_levels, beta)
    Z = Partition(energy_levels, beta)
    return sum(energy_levels.*exp.(-energy_levels.*beta))/Z
end

function Squared_Energy(energy_levels, beta)
    Z = Partition(energy_levels, beta)
    return sum((energy_levels.^2).*exp.(-energy_levels.*beta))/(Z)
end

function Prob(energy_level, beta)
    Z = Partition(energy_levels, beta)
    return exp(-energy_level*beta)/Z
end

function Shannon_Entropy(energy_levels, beta)
    Z = Partition(energy_levels, beta)
    probs = exp.(-energy_levels.*beta)/Z
    S = -sum(probs.*log.(probs))
    return S
end