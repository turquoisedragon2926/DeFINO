
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

# Set the event number
nev = 40

# Define the output directory and ensure it exists
output_dir = "num_ev_$(nev)"
# mkpath(output_dir)

# Loop over the range of file indices
for i in 1:1
    # Construct file paths
    input_file = "num_ev_40/states_sample_$(i).jld2"
    eigvec_file = "num_ev_40/FIM_eigvec_sample_$(i).jld2"
    
    # Load the specific variable from the file
    JLD2.@load input_file single_stored_object

    for t in 1:5
        # Extract the saturation value; adjust indexing/keys if needed
        sat = single_stored_object[t]

        figure()
        imshow(reshape(sat, 256, 256)', cmap="jet")
        colorbar(fraction=0.04)
        title("Saturation at time step=$(t)")
        savefig("img_$(nev)/Sample_$(i)_Saturation_$(t)_revised.png")
        close("all")
    end

    # Load the specific variable from the file
    # JLD2.@load eigvec_file single_stored_object

    # for e in 1:1
    #     for t in 1:5
    #         # Extract the saturation value; adjust indexing/keys if needed
    #         sat = single_stored_object[:,:,e,t]

    #         figure()
    #         maxabs_U = maximum(abs, sat)*0.8
    #         linthresh = 0.1 * maxabs_U
    #         norm_U = PyPlot.matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-maxabs_U, vmax=maxabs_U)
    #         imshow(reshape(sat, 256, 256)', cmap="seismic", norm=norm_U)#norm=mcolors.CenteredNorm(0))
    #         colorbar(fraction=0.04)
    #         title("Left Singular Vector $(e) at time step = $(t)")
    #         filename = "img_$(nev)/Sample_10_U_svd_$(t)_$(e).png"
    #         savefig(filename)
    #         close("all")
    #     end
    # end
    
    # # Initialize an array to store saturation data
    # sat_series = []
    # for t in 1:5
    #     # Extract the saturation value; adjust indexing/keys if needed
    #     sat = Saturations(single_stored_object[t][:state])
    #     push!(sat_series, sat)
    # end
    
    # # Save the saturation series to the output file
    # JLD2.@save output_file sat_series
end
