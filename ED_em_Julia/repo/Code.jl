using Pkg
using Combinatorics
using LinearAlgebra
using SparseArrays
using Arpack
using Plots
#using BenchmarkTools
#using CUDA
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#using DataFrames
#using CSV

include("ExactDiag.jl")
#include("Termody.jl")

Nsitios = 8

indices = Base_ind(Nsitios)
indices_sim = Symmetries(true, true, Nsitios%2, Nsitios, indices)

Hdim = length(indices_sim)

function Hubbard(ed_, t_, U_)
    Hmu = Honsite(Nsitios, indices_sim, ed_)
    Ht = Hhopping(Nsitios, indices_sim, t_)
    HU = Hint(Nsitios, indices_sim, U_)

    H = Hmu + Ht + HU + U_*I./2

    Ham = Matrix(H)
    data = eigen(Ham)

    return data.values, data.vectors

end

E, Vecs = Hubbard(0.5, 1, 1)

print(E, "\n\n")
