using Pkg; Pkg.activate(".")

# Set up threading for BLAS, etc.
nthreads = try
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch e
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)
println("Number of threads: ", Threads.nthreads())

using JutulDarcyRules
using Random
Random.seed!(2023)
using PyPlot
using PyCall
@pyimport matplotlib.colors as mcolors
using JLD2
using Printf
using Statistics
using Distributions
using Zygote
using Distributed
using Interpolations

# ------------------ #
# Load Dataset       #
# ------------------ #
JLD2.@load "../../../../Diff_MultiPhysics/FNO-NF.jl/scripts/wise_perm_models_2000_new.jld2"  # loads BroadK, phi, md, etc.

# ------------------ #
# Setting            #
# ------------------ #

## grid size
dx = 256
dy = 256
n = (dx, 1, dy)
d = (15.0, 10.0, 15.0)  # in meters

# rescale permeability
function resize_array(A, new_size)
    itp = interpolate(A, BSpline(Linear()))
    etp = extrapolate(itp, Line())
    xs = [range(1, stop=size(A, i), length=new_size[i]) for i in 1:ndims(A)]
    return [etp(x...) for x in Iterators.product(xs...)]
end

BroadK_rescaled = resize_array(BroadK, (size(BroadK)[1], dx, dy))
figure()
imshow(reshape(BroadK_rescaled[1, :, :], n[1], n[end])', cmap="viridis")
scatter(130, 205, color="red")
colorbar(fraction=0.04)
title("Permeability Model")
savefig("Downsampled_Perm.png")
close("all")

# permeability
K_all = md * BroadK_rescaled
println("size K_all: ", size(K_all))

# Define JutulModel
phi_rescaled = resize_array(phi, (dx, dy))
ϕ = phi_rescaled
top_layer = 70   # Q: top layer?
h = (top_layer - 1) * 1.0 * d[end]
model = jutulModel(n, d, vec(ϕ), K1to3(K_all[1, :, :]; kvoverkh=0.36), h, false)

## simulation time steppings
tstep = 80 * ones(1)  # in days
tot_time = sum(tstep)

## injection & production
inj_loc = (130, 1, 205) .* d
irate = 5e-3
q = jutulSource(irate, [inj_loc])  # injection

# set up modeling operator
S = jutulModeling(model, tstep)

# ------------------ #
# Settings for FIM   #
# ------------------ #
nsample = 400
nev = 20   # Number of eigenvalues/eigenvectors to compute
nt = length(tstep)
μ = 0.0    # Mean of noise
σ = 1.0    # Std of noise
dist = Normal(μ, σ)

# ---------------------- #
# Distributed Setup    #
# ---------------------- #
if nprocs() == 1
    addprocs(5)
end
println("num procs: ", nprocs())

# Load packages on all workers (split the colon import into a separate line)
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules, PyPlot, PyCall, Interpolations, JLD2, Printf
@everywhere using Statistics: mean, std
@everywhere @pyimport matplotlib.colors as mcolors

@everywhere function save_obj(filename::String, obj)
    JLD2.@save filename obj
end


# Define helper functions on all workers
@everywhere function compute_pullback(Fp, col_U)
    try
        return Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ", e)
        return zeros(length(col_U))  # Return zeros on failure
    end
end

@everywhere function compute_gradient_with_pullback(j, cur_state_sat, pb, σ, dist)
    println("perturbation $(j)")
    noise = rand(dist, size(cur_state_sat))
    noise ./= maximum(abs.(noise))
    grad = nothing
    try
        grad = pb(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $(j): ", e)
        grad = zeros(length(cur_state_sat))
    end
    return grad
end

# Wrap the work for one sample in a function
@everywhere function process_sample(i, K_all, n, d, ϕ, h, tstep, q, dist, σ, nev)
    println("Processing sample $(i)")
    # Extract sample K from the overall permeability array
    K = K_all[i, :, :]

    # Update model and modeling operator for this sample
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, false)
    S = jutulModeling(model, tstep)

    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state  # state0 is a Dict with state info
    states = []  # To store simulation states

    # Prepare arrays to store FIM results for 5 time steps
    eigvec_save = zeros(n[1], n[end], nev, 5)
    one_Jvs = zeros(n[1]*n[end], nev, 5)

    # Loop over time steps (example: 5 steps)
    for time_step in 1:5
        println("Sample $(i) - time step $(time_step)")
        # Define state and extraction functions for this time step.
        # Here we assume the state dictionary has a key :Saturations.
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = state(x)[:state][:Saturations]

        # Update simulator variables
        cur_state = state(K)
        # Assume that cur_state[:state] is a dictionary containing :Saturations
        state0_temp = deepcopy(cur_state[:state])
        cur_state_sat = cur_state[:state][:Saturations]

        # Save a saturation figure (optional)
        figure()
        imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
        colorbar(fraction=0.04)
        title("Saturation at sample $(i) time step=$(time_step)")
        savefig("Saturation_$(i)_$(time_step).png")
        close("all")

        push!(states, cur_state)  # Store the current state

        # ------------ #
        # Compute FIM  #
        # ------------ #
        # Compute the pullback function from state_sat.
        Fv, Fp = Zygote.pullback(state_sat, vec(K))
        dll = zeros(n[1]*n[end], nev)

        # Compute gradients for each eigenvalue/component in parallel over perturbations.
        gradient_results = pmap(j -> compute_gradient_with_pullback(j, cur_state_sat, Fp, σ, dist), 1:nev)
        temp = hcat(gradient_results...)
        dll .= temp

        # Compute SVD of the "Jacobian" matrix
        U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        figure()
        imshow(reshape(U_svd[:, 1], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
        colorbar(fraction=0.04)
        title("Largest LSV sample $(i) time step=$(time_step)")
        savefig("U_svd_$(i)_$(time_step).png")
        close("all")

        # Compute Jacobian vector products in parallel for each eigen-component.
        Jv_results = pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)
        figure()
        imshow(reshape(Jv_results[:, 1], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
        colorbar(fraction=0.04)
        title("Jacobian Vector Products sample $(i) time step=$(time_step)")
        savefig("vjp_$(i)_$(time_step).png")
        close("all")

        for e in 1:nev
            one_Jvs[:, e, time_step] = Jv_results[e]
        end

        # Update state0 for next time step.
        state0 = deepcopy(state0_temp)
    end

    # Save outputs for this sample.
    save_obj("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_obj("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_obj("num_ev_$(nev)/states_sample_$(i).jld2", states)
    return true
end

# ---------------------- #
# Run the parallel loop#
# ---------------------- #
results = pmap(i -> process_sample(i, K_all, n, d, ϕ, h, tstep, q, dist, σ, nev), 1:nsample)
