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
tstep = 150 * ones(1) #in days
tot_time = sum(tstep)

## injection & production
inj_loc_idx = (130, 1 , 205) #(145, 1 , 236)
inj_loc = inj_loc_idx .* d
irate = 8e-3
q = jutulSource(irate, [inj_loc]) # injection
# set up modeling operator
S = jutulModeling(model, tstep)


figure()
imshow(reshape(BroadK_rescaled[1, :, :], n[1], n[end])', cmap="viridis")
scatter(inj_loc_idx[1], inj_loc_idx[3], color="red")
colorbar(fraction=0.04)
title("Permeability Model")
savefig("Downsampled_Perm.png")
close("all")


# ------------------ #
# Setting for FIM    #
# ------------------ #


nsample = 25
nev = 8  # Number of eigenvalues and eigenvectors to compute
nt = length(tstep)
μ = 0.0   # Mean of the noise
σ = 1.0   # Standard deviation of the noise
dist = Normal(μ, σ)


# ---------------------- #
# Generate Joint Samples #
# ---------------------- #

if nprocs() == 1
    addprocs(8, exeflags=["--threads=2"])
end
println("num procs: ", nprocs())  # Check total number of processes (should be CPU_THREADS)

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

@everywhere function compute_pullback(Fp, col_U)
    try
        println("col_U", size(col_U))
        return @time Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ", e)
        return zeros(length(col_U))  # Return zeros on failure
    end
end

@everywhere function compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
    println("Perturbation $(j) processed on worker $(myid())")
    try
        return @time Fp(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $(j): ", e)
        return zeros(length(cur_state_sat))
    end
end

@everywhere function generate_masked_noise_column(mask, n)
    # Create a zero vector of length n
    v = zeros(n)
    # Fill in the positions given by the mask with random normal values
    v[mask] .= randn(sum(mask)) #size of v = (256*256,)
    v ./= (maximum(abs.(v)))
    return v
end

@everywhere function generate_orthogonal_masked_noise(saturation, shape::Tuple, nev::Int)
    # Define the mask and total number of elements
    mask = (saturation .!= 0)
    n = prod(shape) #size of n = (256*256,)
    
    # Generate each noise column in parallel using pmap
    noise_columns = pmap(_ -> generate_masked_noise_column(mask, n), 1:nev)
    # Combine the noise columns into a matrix R (n x nev)
    R = hcat(noise_columns...)
    # println("sizeR", size(R))
    # Perform QR decomposition so that Q's columns are orthonormal.
    # Because each column of R is zero outside the mask, every column of Q
    # will also live in that subspace.
    Q = @time qr(R).Q
    # println("sizeQ", size(Q)) #  0.004628 seconds (6 allocations: 1.000 MiB) 
    # println("shape", shape)
    # Reshape each orthonormal column back to the desired multidimensional shape
    noise_vectors = [reshape(Q[:, j], shape) for j in 1:nev]
    return noise_vectors
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
        # gradient_results = pmap(j -> compute_gradient_with_pullback(j, cur_state_sat, Fp, σ, dist), 1:nev)
        # Generate nev orthogonal noise vectors for cur_state_sat's shape.
        noise_vectors = @time generate_orthogonal_masked_noise(cur_state_sat, size(cur_state_sat), nev)

        gradient_results = pmap(j -> begin
            noise = noise_vectors[j]
            compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
        end, 1:nev)
     
        # Store results in dll
        dll .= hcat(gradient_results...)
        # println("size temp", size(temp))
        # dll .= temp
        println("size dll", size(dll))

        @time U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        println("size", size(U_svd), size(S_svd),size(VT_svd))
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        # Create a new figure
        if i == 1
            figure()
            semilogy(S_svd, "o-")
            xlabel("Index")
            ylabel("Singular Value")
            title("Singular Value Decay")
            grid(true)
            savefig("img_$(nev)/Sample_$(i)_SingularValue.png")
            close("all")

            for j in 1:nev
                figure()
                imshow(reshape(U_svd[:, j], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
                colorbar(fraction=0.04)
                title("Left Singular Vector $(j) at time step = $(time_step)")
                filename = "img_$(nev)/Sample_$(i)_U_svd_$(time_step)_$(j).png"
                savefig(filename)
                close("all")
            end
        end

        # Parallelize over `e` using `pmap`
        println("Compute vTJ")
        Jv_results = @time pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)
        Jv_matrix = hcat(Jv_results...)
        println(size(Jv_results), size(Jv_matrix))

        if i == 1
            for j in 1:nev
                figure()
                imshow(reshape(Jv_matrix[:, j], n[1], n[end])', cmap="seismic", norm=mcolors.CenteredNorm(0))
                colorbar(fraction=0.04)
                title("Jacobian Vector Products with LSV $(j) at t = $(time_step)")
                filename = "img_$(nev)/Sample_$(i)_vjp_$(time_step)_$(j).png"
                savefig(filename)
                close("all")
            end
        end

        # Store results
        for e in 1:nev
            println("size", size(Jv_matrix[:, e]))
            one_Jvs[:, e, time_step] = Jv_matrix[:, e]
        end
        state0 = deepcopy(state0_temp)
    end
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)
end

