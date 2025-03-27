##
## This file is for saving the rescaled K.
##

using Pkg; Pkg.activate(".")
using HDF5

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
    xs = [range(1, stop=size(A, i), length=new_size[i]) for i in 1:ndims(A)]
    return [etp(x...) for x in Iterators.product(xs...)]
end

BroadK_rescaled = resize_array(BroadK, (size(BroadK)[1], dx, dy))


# permeability
K_all = md * BroadK_rescaled
println("K_all size: ", size(K_all))

# ------------------ #
# Gen Permeability   #
# ------------------ #

# Save the fisrt 200 samples
K_subset = K_all[1:200, :, :]
h5write("rescaled_200_fields.h5", "K_subset", K_subset)


figure()
imshow(reshape(K_all[1, :, :], n[1], n[end])', cmap="viridis")
scatter(130, 205, color="red")
colorbar(fraction=0.04)
title("Permeability Model")
savefig("Downsampled_Perm_1_save.png")
close("all")


# Define JutulModel
phi_rescaled = resize_array(phi, (dx,dy))
ϕ = phi_rescaled
top_layer = 70
h = (top_layer-1) * 1.0 * d[end]  
model = jutulModel(n, d, vec(ϕ), K1to3(K_all[1,:,:]; kvoverkh=0.36), h, true)

## simulation time steppings
tstep = 365 * ones(1) #in days
tot_time = sum(tstep)

## injection & production
inj_loc_idx = (130, 1 , 205)
inj_loc = inj_loc_idx .* d
irate = 7e-3
q = jutulSource(irate, [inj_loc])
S = jutulModeling(model, tstep)

# ------------------ #
# Gen Saturation     #
# # ------------------ #

# nsample = 40
# nev = 8  # Number of eigenvalues and eigenvectors to compute
# nt = length(tstep)

# for i = 21:nsample
#     Base.flush(Base.stdout)

#     println("sample $(i)")
#     K = K_all[i, :, :]

#     # 0. update model
#     model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, true)
#     S = jutulModeling(model, tstep)

#     # 1. compute forward: input = K
#     mesh = CartesianMesh(model)
#     logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
#     state00 = jutulSimpleState(model)
#     state0 = state00.state  # 7 fields
#     states = []

#     # Repeat for 5 time steps
#     for time_step in 1:5
#         println("Sample $(i) time step $(time_step)")
#         state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
#         state_sat(x) = Saturations(state(x)[:state])

#         cur_state = state(K)
#         state0_temp = deepcopy(cur_state[:state])
#         cur_state_sat = Saturations(cur_state[:state])

#         figure()
#         imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
#         colorbar(fraction=0.04)
#         title("Saturation at time step=$(time_step)")
#         savefig("img_$(nev)/Sample_$(i)_Saturation_$(time_step)_re.png")
#         close("all")
# 
#         push!(states, cur_state_sat)

#         state0 = deepcopy(state0_temp)
#     end
#     save_object("num_ev_$(nev)_stateonly/states_sample_$(i).jld2", states)
#     GC.gc()
# end