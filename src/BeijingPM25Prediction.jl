"""
BeijingPM25Prediction.jl

A Julia package for predicting Beijing PM2.5 concentrations using Neural ODEs.
Implements the data-driven approach: dC/dt = MLP(C, W)
"""
module BeijingPM25Prediction

using CSV, DataFrames, Statistics, Plots, Random, Flux, Dates

export run_pm25_experiment, analyze_results, plot_predictions

"""
    run_pm25_experiment()

Run the complete Neural ODE experiment for Beijing PM2.5 prediction.
Tests 1-hour, 24-hour, and 168-hour prediction horizons.

Returns a dictionary with results for each prediction horizon.
"""
function run_pm25_experiment()
    Random.seed!(42)
    
    println(" Beijing PM2.5 Neural ODE Experiment")
    println("="^50)
    
    # Ensure results directory exists
    mkpath("results/figures")
    
    # Load and process data
    files = filter(x -> startswith(x, "PRSA_Data_"), readdir("data/raw"))
    if isempty(files)
        error("No PRSA data files found in data/raw/")
    end
    
    df = CSV.read(joinpath("data/raw", files[1]), DataFrame)
    
    # Data preprocessing
    pm25_data = [val == "NA" ? NaN : parse(Float64, val) for val in df."PM2.5"]
    temp_data = [val == "NA" ? NaN : parse(Float64, val) for val in df.TEMP]
    pres_data = [val == "NA" ? NaN : parse(Float64, val) for val in df.PRES]
    
    raw_data = hcat(pm25_data, temp_data, pres_data)
    valid_rows = [!any(isnan.(row)) for row in eachrow(raw_data)]
    clean_data = Float32.(raw_data[valid_rows, :])
    
    data_mean = mean(clean_data, dims=1)
    data_std = std(clean_data, dims=1) .+ 1f-8
    norm_data = (clean_data .- data_mean) ./ data_std
    
    println(" Data processed: $(size(norm_data)) samples")
    
    # Multi-horizon experiments
    horizons = [1, 24, 168]  # 1 hour, 1 day, 1 week
    all_results = Dict()
    
    for horizon in horizons
        println("\\n📈 Training $(horizon)-hour prediction model...")
        
        # Create sequences
        seq_len = 24
        n_samples = size(norm_data, 1) - seq_len - horizon
        
        X = zeros(Float32, seq_len, 3, n_samples)
        y = zeros(Float32, n_samples)
        
        for i in 1:n_samples
            X[:, :, i] = norm_data[i:i+seq_len-1, :]
            y[i] = norm_data[i+seq_len+horizon-1, 1]
        end
        
        # Train-test split
        n_train = floor(Int, 0.8 * n_samples)
        X_train, y_train = X[:, :, 1:n_train], y[1:n_train]
        X_test, y_test = X[:, :, n_train+1:end], y[n_train+1:end]
        
        println("  Data split: train=$(n_train), test=$(length(y_test))")
        
        # Neural ODE model: dC/dt = MLP(C, W)
        model = Chain(Dense(3, 32, tanh), Dense(32, 16, tanh), Dense(16, 1))
        optimizer = Flux.Adam(0.01f0)
        opt_state = Flux.setup(optimizer, model)
        
        # Training loop
        for epoch in 1:30
            total_loss = 0.0f0
            for i in 1:n_train
                current_state = X_train[end, :, i]
                target = y_train[i]
                
                loss, grads = Flux.withgradient(model) do m
                    derivative = m(current_state)[1]
                    predicted = current_state[1] + Float32(horizon) * derivative
                    (predicted - target)^2
                end
                
                Flux.update!(opt_state, model, grads[1])
                total_loss += loss
            end
            
            if epoch % 15 == 0
                avg_loss = total_loss / n_train
                println("  Epoch $(epoch): Loss = $(round(Float64(avg_loss), digits=6))")
            end
        end
        
        # Prediction and evaluation
        predictions = Float32[]
        for i in 1:length(y_test)
            current_state = X_test[end, :, i]
            derivative = model(current_state)[1]
            predicted = current_state[1] + Float32(horizon) * derivative
            push!(predictions, predicted)
        end
        
        # Denormalize results
        pred_real = predictions .* data_std[1] .+ data_mean[1]
        target_real = y_test .* data_std[1] .+ data_mean[1]
        
        # Calculate metrics
        mae = mean(abs.(pred_real .- target_real))
        rmse = sqrt(mean((pred_real .- target_real).^2))
        r2 = cor(pred_real, target_real)^2
        
        all_results[horizon] = (
            mae=mae, rmse=rmse, r2=r2, 
            predictions=pred_real, targets=target_real,
            model=model
        )
        
        println("  Results: MAE=$(round(Float64(mae), digits=2)) μg/m³, R²=$(round(Float64(r2), digits=4))")
        
        # Generate and save plots
        plot_predictions(pred_real, target_real, horizon, r2)
    end
    
    # Generate summary plots
    generate_summary_plots(all_results)
    
    println("\\n Experiment completed successfully!")
    println(" Results saved to results/figures/")
    
    return all_results
end

"""
    plot_predictions(predictions, targets, horizon, r2)

Create scatter plot of predictions vs targets for a given horizon.
"""
function plot_predictions(predictions, targets, horizon, r2)
    p = scatter(targets, predictions, alpha=0.6, markersize=2,
               xlabel="True PM2.5 (μg/m³)", ylabel="Predicted PM2.5 (μg/m³)",
               title="Neural ODE: dC/dt = MLP(C,W)\\n$(horizon)h Prediction, R² = $(round(Float64(r2), digits=3))",
               legend=false, color=:blue)
    
    min_val = min(minimum(targets), minimum(predictions))
    max_val = max(maximum(targets), maximum(predictions))
    plot!(p, [min_val, max_val], [min_val, max_val], line=:dash, color=:red, linewidth=2)
    
    filename = "results/figures/neural_ode_$(horizon)h_prediction.png"
    savefig(p, filename)
    println("   Saved plot: $(filename)")
    
    return p
end

"""
    generate_summary_plots(results)

Generate comprehensive summary plots comparing all prediction horizons.
"""
function generate_summary_plots(results)
    horizons_list = sort(collect(keys(results)))
    maes = [results[h].mae for h in horizons_list]
    rmses = [results[h].rmse for h in horizons_list]
    r2s = [results[h].r2 for h in horizons_list]
    
    # Performance comparison plots
    p1 = bar(string.(horizons_list), maes, title="MAE by Prediction Horizon",
             xlabel="Hours", ylabel="MAE (μg/m³)", legend=false, color=:orange)
    
    p2 = bar(string.(horizons_list), rmses, title="RMSE by Prediction Horizon",
             xlabel="Hours", ylabel="RMSE (μg/m³)", legend=false, color=:green)
    
    p3 = bar(string.(horizons_list), r2s, title="R² by Prediction Horizon", 
             xlabel="Hours", ylabel="R²", legend=false, ylims=(0,1), color=:purple)
    
    # Time series example
    sample_data = results[1]
    n_show = min(200, length(sample_data.targets))
    p4 = plot(1:n_show, [sample_data.targets[1:n_show] sample_data.predictions[1:n_show]], 
             label=["True" "Predicted"], linewidth=2,
             xlabel="Time Steps", ylabel="PM2.5 (μg/m³)",
             title="1h Prediction Time Series")
    
    # Save combined plots
    summary1 = plot(p1, p2, layout=(1, 2), size=(800, 400))
    savefig(summary1, "results/figures/mae_rmse_comparison.png")
    
    summary2 = plot(p3, p4, layout=(1, 2), size=(800, 400))
    savefig(summary2, "results/figures/r2_timeseries_comparison.png")
    
    full_summary = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    savefig(full_summary, "results/figures/complete_experiment_summary.png")
    
    println(" Summary plots generated")
end

"""
    analyze_results(results)

Print detailed analysis of experimental results.
"""
function analyze_results(results)
    println("\\n Detailed Results Analysis:")
    println("="^60)
    
    for horizon in sort(collect(keys(results)))
        r = results[horizon]
        println("$(horizon) hours prediction:")
        println("  MAE:  $(round(Float64(r.mae), digits=2)) μg/m³")
        println("  RMSE: $(round(Float64(r.rmse), digits=2)) μg/m³") 
        println("  R²:   $(round(Float64(r.r2), digits=4))")
        println("  Samples: $(length(r.predictions))")
        println()
    end
end

end  # module
