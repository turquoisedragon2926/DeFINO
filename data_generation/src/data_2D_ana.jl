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
using Interpolations


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
function resize_array(A, new_size)
    itp = interpolate(A, BSpline(Linear()))  # Linear interpolation
    etp = extrapolate(itp, Line())          # Extrapolation method
    scale_factors = [(new_size[i] - 1) / (size(A, i) - 1) for i in 1:ndims(A)]
    
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

# BroadK_rescaled = BroadK[:, 1:dx, 256-dy+1:256]
# permeability
K_all = md * BroadK_rescaled
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

addprocs(5)  # Leave 1 core for the main process
println("num procs: ", nprocs())  # Check total number of processes (should be CPU_THREADS)

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 


@everywhere function compute_gradient_with_pullback(j, cur_state_sat, pb, σ, dist)
    println("perturbation $(j)")
    # Generate noise and normalize it
    noise = rand(dist, size(cur_state_sat))
    noise ./= (maximum(abs.(noise)) * 2.0)
    
    # Create the perturbed input from the current state saturation
    perturbed_input = vec(cur_state_sat + noise)
    
    # Instead of re-calling pullback, use the precomputed F and pb.
    # The loss is:  L(x) = || perturbed_input - state_sat(x) ||^2/(2σ^2)
    # Its gradient at x = vec(K) is given by:
    #    dL/dx = pb((state_sat(vec(K)) - perturbed_input)/σ^2)[1]
    grad = nothing
    try
        grad = pb(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $(j): ", e)
        grad = zeros(length(perturbed_input))
    end
    
    println("outvec $(j)")
    return grad
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
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, false)
    # model = jutulModel(n, d, ϕ, K1to3(K))
    S = jutulModeling(model, tstep)

    # 1. compute forward: input = K
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state0 = jutulSimpleState(S.model)
    states = []

    # Repeat for 8 time steps
    for time_step in 1:5
        println("time step $(time_step)")
        if time_step == 1
            state(x) = S(logTrans(x), q, info_level=1)[1]
        else
            state(x) = S(logTrans(x), q, state0=state0, info_level=1)[1]
        end
        state_sat(x) = Saturations(state(x)[:state])
        pressure(x) = Pressure(state(x)[:state])

        # update simulator variables
        state0_temp = deepcopy(state0)
        cur_state = state(K) 
        cur_state_sat = Saturations(cur_state[:state])

        figure()
        imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
        colorbar(fraction=0.04)
        title("Saturation at time step=$(time_step)")
        savefig("Saturation_$(time_step).png")

        if time_step == 1
            state0_temp.state[:Saturations] = cur_state_sat
            state0_temp.state[:Pressure] = pressure(K)
        else
            state0_temp[:state][:Saturations] = cur_state_sat
            state0_temp[:state][:Pressure] = pressure(K)
        end
        push!(states, cur_state)  # Store the current state in the states array

        # ------------ #
        # Compute FIM  #
        # ------------ #

        # Compute the pullback function
        Fv, Fp = Zygote.pullback(state_sat, vec(K))

        dll = zeros(n[1]*n[end], nev)
        # Run gradient computations in parallel using Distributed.jl (j, cur_state_sat, pb, σ, dist)
        gradient_results = pmap(j -> compute_gradient(j, cur_state_sat, Fp, σ, dist), 1:nobs)
        temp = hcat(gradient_results...)
        println("size temp", size(temp))
        dll .= temp

        U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

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
end

#####################


nsample = 1
nobs = 1
# # Number of eigenvalues and eigenvectors to compute
nev = 5
tstep = dt * ones(1) #in days
nt = length(tstep)
μ = 0.0   # Mean of the noise
σ = 1.0   # Standard deviation of the noise
dist = Normal(μ, σ)

using LinearAlgebra, Random, ForwardDiff, Distributed
# addprocs(2)  # Uncomment or adjust as needed

"""
    randomized_svd_state_sat(Kvec; k, p=5, eps=1e-6)

Computes an approximation of the first `k` left singular vectors of the Jacobian of
`state_sat` (evaluated at `Kvec`) using randomized probing. An oversampling of `p`
columns is used for accuracy, and `eps` is the finite-difference step size.
"""

@everywhere using Zygote

# Define the distributed pullback helper on all workers.
@everywhere function compute_pullback(Fp, col_U)
    try
        return Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ", e)
        return zeros(length(col_U))
    end
end

function randomized_svd_state_sat(Kvec, state_sat; k, p=5, eps=1e-7)
    # Precompute the baseline state at Kvec.
    U0 = state_sat(Kvec)
    m = length(U0)      # Dimension of the output (state_sat)
    n = length(Kvec)    # Dimension of the input (Kvec)

    println("Computing Orthonormal Bases")
    nonzero_idx = findall(x -> x != 0, U0)
    # Step 2: Generate a random vector on that support.
    r = randn(length(nonzero_idx))
    # Step 3: Normalize it to create a unit vector.
    r = r / norm(r)
    
    # Step 4: Create a full vector with zeros and assign the random values at the nonzero positions.
    v = zeros(length(U0))
    v[nonzero_idx] = r
    Q = repeat(v, 1, k+p)
    println("Computed Orthonormal Bases")
    println("size of Q: ", size(Q))

    # --- Compute the pullback once on the main process ---
    Fv, Fp = Zygote.pullback(state_sat, Kvec)
    
    # --- Form the projected matrix B = Qᵀ * J ---
    # Use pmap to apply the pullback (via compute_pullback) to each column of Q.
    B_rows = pmap(idx -> compute_pullback(Fp, Q[:, idx]), 1:(k+p))
    # Combine the results (each returned vector is one row of B).
    B = hcat(B_rows...)'  # hcat makes columns; transpose to get rows

    for idx in 1:(k+p)
        println("idx ", idx)
        println("B row: ", B[idx, 1:min(end,10)])
    end

    # --- Compute the SVD of the small matrix B ---
    svdB = svd(B)
    U_B = svdB.U[:, 1:k]      # Keep only the first k columns.
    S_vals = svdB.S[1:k]
    Vt = svdB.Vt[1:k, :]

    # --- Approximate the left singular vectors of J ---
    # The approximate left singular vectors are given by Q * U_B.
    U_approx = Q * U_B

    return U_approx, S_vals, Vt
end

# === Example usage ===
k = 5
Kvec = vec(K_resized)

# Precompute constant for the simulator
mesh = CartesianMesh(model)
Trans(x) = KtoTrans(mesh, K1to3(x; kvoverkh=0.36*one(eltype(x))))
ϕ_pad = vec(padϕ(ϕ_resized))  # precomputed to avoid in-place mutation
q = jutulVWell(0.03, (Float64(inj_loc[1]), Float64(inj_loc[2]));
    startz = Float64(inj_loc[3]), endz = Float64(inj_loc[3] + 2.0))

# Define Simulator
println("Defined Simulator")
state(x) = S(log.(Trans(x)), ϕ_pad, q, info_level=1)[1]
state_sat(x) = Saturations(state(x)[:state])

# Compute the pullback from state_sat at Kvec (for diagnostic purposes).
println("Calling pullback")
U, dUdK = Zygote.pullback(state_sat, Kvec)

println("Randomized SVD")
U_approx, singular_vals, Vt = randomized_svd_state_sat(Kvec, state_sat; k=k, p=5, eps=1e-6)

println("Approximate left singular vectors (columns of U_approx):")
println(U_approx)
