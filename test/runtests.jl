using Test
using BeijingPM25Prediction

@testset "BeijingPM25Prediction.jl Tests" begin
    @testset "Module Loading" begin
        @test isdefined(BeijingPM25Prediction, :run_pm25_experiment)
        @test isdefined(BeijingPM25Prediction, :plot_predictions)
        @test isdefined(BeijingPM25Prediction, :analyze_results)
    end
    
    @testset "Function Types" begin
        @test typeof(run_pm25_experiment) == Function
        @test typeof(plot_predictions) == Function
        @test typeof(analyze_results) == Function
    end
end

println("All tests passed!")
