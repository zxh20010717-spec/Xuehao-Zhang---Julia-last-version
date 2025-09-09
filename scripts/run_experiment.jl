#!/usr/bin/env julia
# Beijing PM2.5 Neural ODE Experiment Runner
# Usage: julia --project=. scripts/run_experiment.jl

using Pkg
Pkg.activate(".")

println("🚀 Beijing PM2.5 Neural ODE Experiment")
println("Loading BeijingPM25Prediction package...")

using BeijingPM25Prediction

# Check if data exists
if !isdir("data/raw") || isempty(readdir("data/raw"))
    println("❌ Error: No data found in data/raw/")
    println("Please ensure PRSA CSV files are in the data/raw/ directory")
    exit(1)
end

try
    # Run the complete experiment
    println("Starting experiment...")
    results = run_pm25_experiment()
    
    # Analyze and display results
    analyze_results(results)
    
    println("\\n🎉 Experiment completed successfully!")
    println("📊 Check results/figures/ for generated plots")
    println("📋 Results summary:")
    
    for (horizon, result) in sort(collect(results))
        println("  $(horizon)h: MAE=$(round(Float64(result.mae), digits=2)) μg/m³, R²=$(round(Float64(result.r2), digits=4))")
    end
    
catch e
    println("❌ Experiment failed with error: $e")
    rethrow(e)
end
