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
tstep = 100 * ones(1)  # in days
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
nev = 40   # Number of eigenvalues/eigenvectors to compute
nt = length(tstep)
μ = 0.0    # Mean of the noise
σ = 1.0    # Standard deviation of the noise
dist = Normal(μ, σ)

println("num procs: ", nprocs())

@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

# Define pullback functions without extra logging/timing
@everywhere function compute_pullback(Fp, col_U)
    try
        return Fp(col_U)[1]
    catch e
        return zeros(length(col_U))
    end
end

function compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
    try
        println("Pullback j", j)
        out = @time Fp(noise)[1]
        return out
    catch e
        println("Failed pullback")
    end
end

# ------------------------------- #
# Main sample loop and FIM compute#
# ------------------------------- #
for i in 10:nsample
    # Pre-allocate arrays for saving results
    eigvec_save = zeros(n[1], n[end], nev, 5)
    one_Jvs = zeros(n[1]*n[end], nev, 5)
    states = Vector{Any}()
    
    println("Processing sample $(i)")
    K = K_all[i, :, :]
    
    # Update model for current sample
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, true)
    S = jutulModeling(model, tstep)
    
    # Set up forward model
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state  # initial state
    
    for time_step in 1:5
        println("Sample $(i) time step $(time_step)")
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])
        
        # Compute the forward state and the pullback operator via Zygote
        println("Forward")
        cur_state = state(K)
        println("Backward")
        cur_state_sat, Fp = Zygote.pullback(state_sat, vec(K))
        state0_temp = copy(cur_state[:state])
        
        # Save saturation state for later use
        push!(states, cur_state_sat)
        
        # ----------------------- #
        # Compute the FIM (using noise perturbations)
        # ----------------------- #
        noise_vectors = rand(dist, (n[1]*n[end], nev))
        gradient_results = Vector{Vector{Float64}}(undef, nev)
        
        # Use multi-threading for computing gradient results
        println("Computing pullback")
        Threads.@threads for j in 1:nev
            gradient_results[j] = compute_pullback_with_noise(j, Fp, noise_vectors[:, j], cur_state_sat)
        end
        
        # Build the perturbation matrix (each column is a gradient vector)
        dll = hcat(gradient_results...)
        
        # Compute the SVD of the perturbation matrix
        println("Computing SVD")
        U_svd, S_svd, VT_svd = svd(dll)
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)
        
        # ----------------------- #
        # Compute vTJ (Jacobian-vector product)
        # ----------------------- #
        println("Computing vTJ")
        Jv_matrix = zeros(size(dll))
        Threads.@threads for e in 1:nev
            Jv_matrix[:, e] = compute_pullback(Fp, U_svd[:, e])
        end
        
        for e in 1:nev
            one_Jvs[:, e, time_step] = Jv_matrix[:, e]
        end
        
        # Update state for the next time step
        state0 = copy(state0_temp)
    end
    
    # Save results for current sample
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)
end
