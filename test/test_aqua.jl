using Aqua: Aqua
using NamedDimsArrays: NamedDimsArrays
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # TODO: fix and re-enable ambiguity checks
    Aqua.test_all(NamedDimsArrays; ambiguities = false)
end
