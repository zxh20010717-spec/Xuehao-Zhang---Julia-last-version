using Test
using BeijingPM25Prediction

@testset "basic" begin
    @test isdefined(Main, :BeijingPM25Prediction)
end
