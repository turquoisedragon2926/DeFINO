## A 2D example
using Pkg; Pkg.activate(".")

nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch e
    # Desktop
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)
println("Number of threads:", Threads.nthreads())

using JutulDarcyRules
using Random
Random.seed!(2023)
using PyPlot
using JLD2
using Printf
using Statistics: mean, std
using Distributions
using Zygote
using Distributed


# ------------------ #
# Load Dataset       #
# ------------------ #

JLD2.@load "../../../../Diff_MultiPhysics/FNO-NF.jl/scripts/wise_perm_models_2000_new.jld2" #phi = porosity

# ------------------ #
# Setting            #
# ------------------ #

## grid size
dx = 512
dy = 256
n = (dx, 1, dy)
d = (15.0, 10.0, 15.0) # in meters

# rescale permeability
# BroadK_rescaled = BroadK[:, 1:dx, 256-dy+1:256]
# permeability
K_all = md * BroadK
print(size(K_all))

# Define JutulModel
ϕ = 0.25
# ϕ = 0.25 * ones(n)
model = jutulModel(n, d, ϕ, K1to3(Matrix(K_all[1,:,:])))

## simulation time steppings
tstep = 80 * ones(1) #in days
tot_time = sum(tstep)

## injection & production
inj_loc = (130, 1, 205) .* d
irate = 5e-3
q = jutulSource(irate, [inj_loc]) # injection
# set up modeling operator
S = jutulModeling(model, tstep)


# ------------------ #
# Setting for FIM    #
# ------------------ #


nsample = 400
nobs = 20
# # Number of eigenvalues and eigenvectors to compute
nev = 20
nt = length(tstep)
μ = 0.0   # Mean of the noise
σ = 1.0   # Standard deviation of the noise
dist = Normal(μ, σ)


# ---------------------- #
# Generate Joint Samples #
# ---------------------- #

# addprocs(2)  # Leave 1 core for the main process
println("num procs: ", nprocs())  # Check total number of processes (should be CPU_THREADS)

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

# Define function on all workers
@everywhere function compute_gradient(j, cur_state_sat, state_sat, dist, σ, K)
    println("perturbation $(j)")
    noise = rand(dist, size(cur_state_sat))
    noise ./= (maximum(abs.(noise)) * 2.0)

    perturbed_input = vec(cur_state_sat + noise)

    ll(x) = norm(perturbed_input - state_sat(x))^2 / (2 * σ^2)
    # dll(x) = 2 * (perturbed_input - state_sat(x))

    outer_vector = nothing
    try
        outer_vector = Zygote.gradient((x) -> ll(x), vec(K))[1]
    catch e
        println("Gradient computation failed for perturbation $(j): ", e)
        outer_vector = zeros(length(vec(K)))  # Assign zeros on failure
    end

    println("outvec", j)
    return outer_vector
end

# Define function for parallel computation
@everywhere function compute_pullback(Fp, col_U)
    try
        return Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ", e)
        return zeros(length(col_U))  # Return zeros on failure
    end
end

for i = 1:nsample # 13
    Base.flush(Base.stdout)

    Ks = zeros(n[1], n[end], nsample);
    eigvec_save = zeros(n[1], n[end], nev, 8);
    one_Jvs = zeros(n[1]*n[end], nev, 8);
    conc = zeros(n[1], n[end], 1);

    println("sample $(i)")
    K = K_all[i,:,:]

    # 0. update model
    model = jutulModel(n, d, ϕ, K1to3(K))
    S = jutulModeling(model, tstep)

    # 1. compute forward: input = K
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))

    state0 = jutulSimpleState(S.model)
    states = []

    # Repeat for 8 time steps
    for time_step in 1:8
        println("time step $(time_step)")

        state(x) = S(logTrans(x), q, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])
        pressure(x) = state(x)[:state][:Pressure]

        # update simulator variables
        state0_temp = deepcopy(state0)
        cur_state = state(K) 
        cur_state_sat = Saturations(cur_state[:state])

        figure()
        imshow(cur_state_sat', cmap="viridis")
        colorbar()
        title("Saturation at time step=$(time_step)")
        savefig("Saturation_$(time_step).png")

        state0_temp[:state][:Saturations] = cur_state_sat
        state0_temp[:state][:Pressure] = pressure(K)
        push!(states, cur_state)  # Store the current state in the states array

        # ------------ #
        # Compute FIM  #
        # ------------ #

        dll = zeros(n[1]*n[end], nev)
        # Run gradient computations in parallel using Distributed.jl
        gradient_results = pmap(j -> compute_gradient(j, cur_state_sat, state_sat, dist, σ, K), 1:nobs)
        temp = hcat(gradient_results...)
        println("size temp", size(temp))
        dll .= temp

        U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        # Compute the pullback function
        Fv, Fp = Zygote.pullback(state_sat, vec(K))

        # Parallelize over `e` using `pmap`
        Jv_results = pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)

        # Store results
        for e in 1:nev
            println("size lsv", size(U_svd[:, e]))
            println("size", size(Jv_results[e]))
            one_Jvs[:, e, time_step] = Jv_results[e]
        end
        state0 = deepcopy(state0_temp)
    end

    save_object("num_obs_$(nobs)/FIM_eigvec_sample_$(i)_nobs_$(nobs).jld2", eigvec_save)
    save_object("num_obs_$(nobs)/FIM_vjp_sample_$(i)_nobs_$(nobs).jld2", one_Jvs)
    save_object("num_obs_$(nobs)/states_sample_$(i)_nobs_$(nobs).jld2", states)

    # # first eigenvector
    # println("eig vec size", size(v1)) # 4096,1
    # println("K size", size(vec(K)))
    # # compute vector Jacobian product
end

