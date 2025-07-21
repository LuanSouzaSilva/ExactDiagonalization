using Pkg
using Combinatorics
using LinearAlgebra
using SparseArrays
using Arpack
using Plots

include("ExactDiag.jl")
#commutator between A and B
function Commutator(A, B)
    return A*B - B*A
end

#Calculates the lindbladian
function Lindblad(rhot_, H_)
    Unit_cont = -1.0im*Commutator(H_, rhot_) #Unitary contribution
    return Unit_cont
end

#Calculates one time step of the density matrix using a "Euler Method Scheme" (naive approach)
function TimeStep(rhot_, H_, dt_)
    new_rho = rhot_ - Lindblad(rhot_, H_)*dt_
    return new_rho/tr(new_rho) #Ensuring normalization of the density matrix
end

Hmu = Honsite(Nsitios, indices_sim, ed_)
Ht = Hhopping(Nsitios, indices_sim, t_)
HU = Hint(Nsitios, indices_sim, U_)

H = Hmu + Ht + HU + U_*I./2

Ham = Matrix(H)
data = eigen(Ham)