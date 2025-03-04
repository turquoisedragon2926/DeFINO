## A 3D example
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

JLD2.@load "../../Diff_MultiPhysics/FNO-NF.jl/scripts/AspireK_128cube_firstdim_sampleid.jld2"

# ------------------ #
# Helper Functions - #
# ------------------ #

function Ktoϕ(K::T; α=T(48.63057324840764)) where T
    p = Polynomial([-K,2*K,-K, α^2])
    return minimum(real(roots(p)[findall(real(roots(p)).== roots(p))]))
end
    
function padϕ(ϕ::Matrix{T}) where T
    return hcat(vcat(T(1e8)*ones(T, 1, size(ϕ,2)-1), ϕ[2:end-1,1:end-1], T(1e8)*ones(T, 1, size(ϕ,2)-1)),
    T(1e8)*ones(T, size(ϕ,1), 1))
end

function padϕ(ϕ::Array{T, 3}) where T
    rv = copy(ϕ)
    # the following comments assume x,y,z coordinates are longitude (west-east), latitude (north-south), depth.  but it's arbitrary.
    # west side
    rv[1,:,:]   .= 1e8
    # east side
    rv[end,:,:] .= 1e8
    # north side
    rv[:,1,:]   .= 1e8
    # south side
    rv[:,end,:] .= 1e8
    # under side
    rv[:,:,end] .= 1e8
    # leave the top side alone, it's supposed to be caprock
    rv
end


# ------------------ #
# Setting            #
# ------------------ #

## grid size
dx = 30
n = (dx, dx, dx)
d = (12.5f0,12.5f0,12.5f0)
d1 = Float64.(d)
dt = 80 #80;
nt = 1;

# rescale permeability
function resize_array(A, new_size)
    itp = interpolate(A, BSpline(Linear()))  # Linear interpolation
    etp = extrapolate(itp, Line())          # Extrapolation method
    xs = [range(1, stop=size(A, i), length=new_size[i]) for i in 1:ndims(A)]
    return [etp(x...) for x in Iterators.product(xs...)]
end

# permeability
BroadK_rescaled = resize_array(BroadK, (size(BroadK)[1], dx, dy))

K_all = md * BroadK_rescaled
println("K_all size: ", size(K_all))

# porosity
# ϕ = Ktoϕ.(AspireK[index,:,:,:])
# por = ϕ

# Define JutulModel
phi_rescaled = resize_array(phi, (dx,dx,dx))
ϕ = phi_rescaled
top_layer = 70
h = (top_layer-1) * 1.0 * d[end]  
model = jutulModel(n, d, vec(ϕ), K1to3(K_all[1,:,:]; kvoverkh=0.36), h, true)

## simulation time steppings
tstep = 150 * ones(1) #in days
tot_time = sum(tstep)

## injection & production
inj_loc_idx = (130, 1 , 205)
inj_loc = inj_loc_idx .* d
irate = 6e-3
q = jutulSource(irate, [inj_loc])
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
println("num procs: ", nprocs())

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

@everywhere function compute_pullback(Fp, col_U)
    try
        println("col_U size: ", size(col_U))
        return @time Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ", e)
        return zeros(length(col_U))  # Return zeros on failure
    finally
        GC.gc()  # Free memory on this worker after computation
    end
end

@everywhere function compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
    println("Perturbation $(j) processed on worker $(myid())")
    try
        return @time Fp(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $(j): ", e)
        return zeros(length(cur_state_sat))
    finally
        GC.gc()
    end
end

@everywhere function generate_masked_noise_column(mask, n)
    v = zeros(n)
    v[mask] .= randn(sum(mask))
    v ./= maximum(abs.(v))
    return v
end

@everywhere function generate_orthogonal_masked_noise(saturation, shape::Tuple, nev::Int)
    mask = (saturation .!= 0)
    n = prod(shape)
    noise_columns = pmap(_ -> generate_masked_noise_column(mask, n), 1:nev)
    R = hcat(noise_columns...)
    Q = @time qr(R).Q
    noise_vectors = [reshape(Q[:, j], shape) for j in 1:nev]
    return noise_vectors
end

for i = 2:nsample
    Base.flush(Base.stdout)

    Ks = zeros(n[1], n[end], nsample)
    eigvec_save = zeros(n[1], n[end], nev, 8)
    one_Jvs = zeros(n[1]*n[end], nev, 8)
    conc = zeros(n[1], n[end], 1)

    println("sample $(i)")
    K = K_all[i, :, :]

    # 0. update model
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, true)
    S = jutulModeling(model, tstep)

    # 1. compute forward: input = K
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state  # 7 fields
    states = []

    # Repeat for 5 time steps
    for time_step in 1:5
        println("Sample $(i) time step $(time_step)")
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])

        cur_state = state(K)
        @time Fv, Fp = Zygote.pullback(state_sat, vec(K))  # v^TJ pullback
        state0_temp = deepcopy(cur_state[:state])
        cur_state_sat = Saturations(cur_state[:state])

        figure()
        imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
        colorbar(fraction=0.04)
        title("Saturation at time step=$(time_step)")
        savefig("img_$(nev)/Sample_$(i)_Saturation_$(time_step).png")
        close("all")

        push!(states, cur_state)

        # ------------ #
        # Compute FIM  #
        # ------------ #
        dll = zeros(n[1]*n[end], nev)
        noise_vectors = @time generate_orthogonal_masked_noise(cur_state_sat, size(cur_state_sat), nev)

        gradient_results = pmap(j -> begin
            noise = noise_vectors[j]
            compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
        end, 1:nev)
     
        # Free noise_vectors now that they’re used
        noise_vectors = nothing
        GC.gc()

        dll .= hcat(gradient_results...)
        println("size dll", size(dll))

        @time U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        println("size U_svd: ", size(U_svd), " S_svd: ", size(S_svd), " VT_svd: ", size(VT_svd))
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        if i == 2
            figure()
            semilogy(S_svd, "o-")
            xlabel("Index")
            ylabel("Singular Value")
            title("Singular Value Decay at time step = $(time_step)")
            grid(true)
            savefig("img_$(nev)/Sample_$(i)_SingularValue_$(time_step).png")
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

        println("Compute vTJ")
        Jv_results = @time pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)
        Jv_matrix = hcat(Jv_results...)
        println("Jv_matrix size: ", size(Jv_matrix))

        if i == 2
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

        for e in 1:nev
            one_Jvs[:, e, time_step] = Jv_matrix[:, e]
        end

        # Free large arrays after each time step
        dll = nothing
        U_svd = nothing
        S_svd = nothing
        VT_svd = nothing
        Jv_matrix = nothing
        GC.gc()

        state0 = deepcopy(state0_temp)
    end
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)

    # Clean up after each sample
    eigvec_save = nothing
    one_Jvs = nothing
    states = nothing
    GC.gc()
end
