using Pkg; Pkg.activate(".")
# Try to pin threads if possible (from fast code)
try
    using ThreadPinning
    pinthreads(:cores)
    thread_map = ThreadPinning.thread_to_cpu_map()
    for (tid, core) in thread_map
        println("Thread $(tid) is pinned to CPU core $(core)")
    end
catch
    println("Fallback due to ThreadPinning not available.")
end

# Set number of threads
nthreads = try
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)
println("Number of threads: ", Threads.nthreads())

using JutulDarcyRules, Random, PyPlot, PyCall, JLD2, Printf, Statistics, Distributions, Zygote, Distributed, Interpolations
Random.seed!(2023)

@pyimport matplotlib.colors as mcolors

# ------------------ #
# Load Dataset       #
# ------------------ #
JLD2.@load "../../../../Diff_MultiPhysics/FNO-NF.jl/scripts/wise_perm_models_2000_new.jld2" # phi

# ------------------ #
# Settings           #
# ------------------ #
dx = 256
dy = 256
n = (dx, 1, dy)
d = (15.0, 10.0, 15.0)  # in meters

# Rescale permeability using linear interpolation
function resize_array(A, new_size)
    itp = interpolate(A, BSpline(Linear()))
    etp = extrapolate(itp, Line())
    xs = [range(1, stop=size(A, i), length=new_size[i]) for i in 1:ndims(A)]
    return [etp(x...) for x in Iterators.product(xs...)]
end

BroadK_rescaled = resize_array(BroadK, (size(BroadK)[1], dx, dy))
K_all = md * BroadK_rescaled
println("K_all size: ", size(K_all))

# Use a constant porosity for simplicity
ϕ = 0.25 * ones((dx, dy))
top_layer = 70
h = (top_layer - 1) * 1.0 * d[end]
model = jutulModel(n, d, vec(ϕ), K1to3(BroadK_rescaled[1, :, :]; kvoverkh=0.36), h, true)

# Simulation time stepping
tstep = 100 * ones(2)  # in days
tot_time = sum(tstep)

# Injection & production setup
inj_loc_idx = (130, 1, 205)
inj_loc = inj_loc_idx .* d
irate = 9e-3
q = jutulSource(irate, [inj_loc])
S = jutulModeling(model, tstep)

# Plot a permeability image for visual check
figure()
imshow(reshape(BroadK_rescaled[15, :, :], n[1], n[end])', cmap="viridis")
scatter(inj_loc_idx[1], inj_loc_idx[3], color="red")
colorbar(fraction=0.04)
title("Permeability Model")
savefig("Downsampled_Perm_15.png")
close("all")

# ------------------ #
# Settings for FIM   #
# ------------------ #
nsample = 200
nev = 40   # Number of noise directions
μ = 0.0
σ = 1.0
dist = Normal(μ, σ)

println("num procs: ", nprocs())
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

# We assume these functions are defined from your code:
#   state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
#   state_sat(x) = Saturations(state(x)[:state])
#
# In your simulation, state(x)[:state] is a 2×256×256 array, and Saturations() maps it to a 256×256 array.

# To work in a batched framework, we define a batched forward function.
function state_sat_batch(X::Matrix{Float64})
    # Here X is a matrix where each column is a perturbation added to the base input.
    # The base input is our current model parameter vector x0.
    base = vec(K)  # K here is the current sample (should be 256×2×256, so length 131072).
    B = size(X, 2)
    results = Vector{Vector{Float64}}(undef, B)
    for j in 1:B
        # Evaluate state_sat at (base + perturbation).
        results[j] = state_sat(base + X[:, j])
    end
    # Each result is expected to be a vector of length 65,536 (i.e. 256×256).
    return hcat(results...)  # Returns a (65536 x B) matrix.
end

# Define a helper to “reduce” a 2‐channel vector in the domain to a single channel.
# That is, if x is a vector of length 131072, we reshape it to (256,2,256)
# and then average over the second dimension to get a 256×256 array (flattened to 65536).
function reduce_sat(x::Vector{Float64})
    A = reshape(x, n[1], 2, n[end])  # (256, 2, 256)
    return vec(mean(A, dims=2))       # Result is (256,1,256) which we flatten to length 65536.
end

# ------------------------------- #
# Main sample loop and FIM compute#
# ------------------------------- #
for i in 10:nsample
    # Pre-allocate arrays for saving results
    # We expect the left singular vectors (after reduction) to be 256×256.
    eigvec_save = zeros(n[1], n[end], nev, 5)
    # one_Jvs will hold our computed Jacobian–vector products (vTJ), of size 256×256 (flattened 65536) per direction.
    one_Jvs = zeros(n[1]*n[end], nev, 5)
    states = Vector{Any}()
    
    println("Processing sample $(i)")
    K = K_all[i, :, :]  # K is assumed to be of size (256, 2, 256) so that vec(K) is length 131072.
    
    # Update model for current sample.
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, true)
    S = jutulModeling(model, tstep)
    
    # Set up forward model components.
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state  # initial state
    
    for time_step in 1:5
        println("Sample $(i) time step $(time_step)")
        # Define the forward function.
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])
        
        # --- Batched AD for noise perturbations ---
        # Domain dimension: N_in = length(vec(K)) = 131072.
        N_in = length(vec(K))
        # Output dimension: N_out = length(state_sat(vec(K))) should be 256*256 = 65536.
        N_out = length(state_sat(vec(K)))
        
        # We form the zero perturbation matrix for the domain.
        X0 = zeros(Float64, N_in, nev)
        # Get the pullback operator for our batched function.
        cur_state_sat_batch, Fp_batch = Zygote.pullback(state_sat_batch, X0)
        
        # Generate a batch of noise perturbations in the output (saturation) space.
        noise_vectors = rand(dist, (N_out, nev))
        # Apply the batched pullback operator.
        # Since state_sat: R^(131072) -> R^(65536), Fp_batch maps from R^(65536) to R^(131072).
        # Thus, batched_gradient will have size (131072, nev).
        batched_gradient = Fp_batch(noise_vectors)[1]
        
        # Compute SVD on the batched gradient matrix.
        println("Computing SVD (batched)")
        U_svd, S_svd, VT_svd = svd(batched_gradient)
        # U_svd is of size (131072, nev). We need to reduce it to the saturation space.
        # Reshape U_svd to (256, 2, 256, nev) and then average over the 2nd dimension.
        U_reshaped = reshape(U_svd, n[1], 2, n[end], nev)
        U_reduced = dropdims(mean(U_reshaped, dims=2), dims=2)  # now size is (256, 256, nev)
        eigvec_save[:, :, :, time_step] = U_reduced  # Save reduced left singular vectors.
        
        # --- Compute vTJ (Jacobian-vector product) in batch ---
        # In the original (unbatched) version, vTJ is computed as Fp(U) for each U.
        # Here, we first reduce each column of U_svd (which is in R^(131072)) to saturation space.
        U_reduced_full = hcat([reduce_sat(U_svd[:,j]) for j in 1:nev]...)
        # U_reduced_full is now of size (65536, nev).
        # Now apply the pullback operator Fp_batch to U_reduced_full.
        # Fp_batch maps from R^(65536) to R^(131072).
        Jv_full = Fp_batch(U_reduced_full)[1]  # size (131072, nev)
        # We then reduce the result back to the saturation space.
        Jv_matrix = hcat([reduce_sat(Jv_full[:,j]) for j in 1:nev]...)  # should be (65536, nev)
        for e in 1:nev
            one_Jvs[:, e, time_step] = Jv_matrix[:, e]
        end
        
        # Update state for the next time step (adjust as needed).
        state0 = copy(state00.state)
    end
    
    # Save results for current sample.
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)
end
