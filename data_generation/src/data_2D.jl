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
using PyCall
@pyimport matplotlib.colors as mcolors
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
dx = 256
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
close("all")

# BroadK_rescaled = BroadK[:, 1:dx, 256-dy+1:256]
# permeability
K_all = md * BroadK_rescaled
print(size(K_all))

# Define JutulModel
phi_rescaled = resize_array(phi, (dx,dy))
ϕ = phi_rescaled
top_layer = 70 #70 # Q: top layer..?
h = (top_layer-1) * 1.0 * d[end]  
model = jutulModel(n, d, vec(ϕ), K1to3(K_all[1,:,:]; kvoverkh=0.36), h, false)

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
nev = 20  # Number of eigenvalues and eigenvectors to compute
nt = length(tstep)
μ = 0.0   # Mean of the noise
σ = 1.0   # Standard deviation of the noise
dist = Normal(μ, σ)


# ---------------------- #
# Generate Joint Samples #
# ---------------------- #

if nprocs() == 1
    addprocs(5)
end
println("num procs: ", nprocs())  # Check total number of processes (should be CPU_THREADS)

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

function compute_gradient_analytical(cur_state_sat, pb, σ, dist)
    # # Option 1: Generate noise and normalize it
    # noise = rand(dist, size(cur_state_sat))
    # noise ./= (maximum(abs.(noise)))

    # Option 2: compute orthonormal bases
    println("Computing Randomized Orthonormal Bases")
    nonzero_idx = findall(x -> x != 0, cur_state_sat)
    # Step 2: Generate a random vector on that support.
    r = randn(length(nonzero_idx))
    # Step 3: Normalize it to create a unit vector.
    r = r / norm(r)
    # Step 4: Create a full vector with zeros and assign the random values at the nonzero positions.
    noise = zeros(length(cur_state_sat))
    noise[nonzero_idx] = r
    println("Computed Orthonormal Bases")
    println("size of Q: ", size(noise))
    
    ## Instead of re-calling pullback, use the precomputed F and pb.
    ## The loss is:  L(x) = || perturbed_input - state_sat(x) ||^2/(2σ^2)
    ## Its gradient at x = vec(K) is given by:
    ##    dL/dx = pb((state_sat(vec(K)) - perturbed_input)/σ^2)[1]
    grad = nothing
    try
        grad = pb(noise)[1]
        println("size of grad", size(grad)) # why is this 256 x 256?
    catch e
        println("Pullback computation failed: ", e)
        grad = zeros(length(cur_state_sat))
    end
    return grad
end

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
    noise ./= (maximum(abs.(noise)))

    grad = nothing
    try
        grad = pb(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $(j): ")
        grad = zeros(length(cur_state_sat))
    end
    return grad
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
    S = jutulModeling(model, tstep)

    # 1. compute forward: input = K
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state # 7 fields
    states = []

    # Repeat for 5 time steps
    for time_step in 1:5
        println("Sample $(i) time step $(time_step)") 
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])

        # update simulator variables
        cur_state = state(K)
        # Compute the pullback function
        @time Fv, Fp = Zygote.pullback(state_sat, vec(K)) #v^TJ 
        state0_temp = deepcopy(cur_state[:state])
        cur_state_sat = Saturations(cur_state[:state])

        figure()
        imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
        colorbar(fraction=0.04)
        title("Saturation at time step=$(time_step)")
        savefig("img_$(nev)/ Sample_$(i)_Saturation_$(time_step).png")
        close("all")

        push!(states, cur_state)  # Store the current state in the states array

        # ------------ #
        # Compute FIM  #
        # ------------ #

        # @time dll = compute_gradient_analytical(cur_state_sat, Fp, σ, dist)
        # 305.549289 seconds (117.32 M allocations: 644.889 GiB, 11.60% gc time, 0.13% compilation time)
        # size of grad: 256 x 256
        dll = zeros(n[1]*n[end], nev)
        # Run gradient computations in parallel using Distributed.jl 
        gradient_results = pmap(j -> compute_gradient_with_pullback(j, cur_state_sat, Fp, σ, dist), 1:nev)
        # Store results in dll
        temp = hcat(gradient_results...)
        println("size temp", size(temp))
        dll .= temp
        println("size dll", size(dll))

        @time U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        # 0.000357 seconds (20 allocations: 512.766 KiB)
        println("size", size(U_svd))
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        figure()
        imshow(reshape(U_svd[:, 1], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
        colorbar(fraction=0.04)
        title("The Largest Left Singular Vector at time step = $(time_step)")
        savefig("img_$(nev)/Sample_$(i)_U_svd_$(time_step).png")
        close("all")

        # Parallelize over `e` using `pmap`
        # Jv_results = Fp(reshape(U_svd, n[1]*n[end]))[1]

        Jv_results = pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)

        figure()
        imshow(reshape(Jv_results[:, 1], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
        colorbar(fraction=0.04)
        title("Jacobian Vector Products with the Largest LSV at time step = $(time_step)")
        savefig("img_$(nev)/Sample_$(i)_vjp_$(time_step).png")
        close("all")

        # Store results
        for e in 1:nev
            println("size lsv", size(U_svd[:, e]))
            println("size", size(Jv_results[e]))
            one_Jvs[:, e, time_step] = Jv_results[e]
        end
        state0 = deepcopy(state0_temp)
    end
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)
end

