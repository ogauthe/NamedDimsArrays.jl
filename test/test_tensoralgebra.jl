using LinearAlgebra: factorize, lq, norm, qr, svd
using NamedDimsArrays: dename, nameddimsindices, namedoneto
using TensorAlgebra:
  TensorAlgebra,
  contract,
  left_null,
  left_orth,
  left_polar,
  matricize,
  orth,
  polar,
  right_null,
  right_orth,
  right_polar,
  unmatricize
using Test: @test, @testset, @test_broken
elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "TensorAlgebra (eltype=$(elt))" for elt in elts
  @testset "contract" begin
    i = namedoneto(2, "i")
    j = namedoneto(2, "j")
    k = namedoneto(2, "k")
    na1 = randn(elt, i, j)
    na2 = randn(elt, j, k)
    na_dest = contract(na1, na2)
    @test eltype(na_dest) === elt
    @test dename(na_dest, (i, k)) ≈ dename(na1) * dename(na2)
  end
  @testset "matricize" begin
    i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, i, j, k, l)
    na_fused = matricize(na, (k, i) => "a", (j, l) => "b")
    # Fuse all dimensions.
    @test dename(na_fused, ("a", "b")) ≈ reshape(
      dename(na, (k, i, j, l)),
      (dename(length(k)) * dename(length(i)), dename(length(j)) * dename(length(l))),
    )
  end
  @testset "unmatricize" begin
    a, b = namedoneto.((6, 20), ("a", "b"))
    i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, a, b)
    # Split all dimensions.
    na_split = unmatricize(na, "a" => (k, i), "b" => (j, l))
    @test dename(na_split, ("k", "i", "j", "l")) ≈
      reshape(dename(na, ("a", "b")), (dename(k), dename(i), dename(j), dename(l)))
  end
  @testset "qr/lq" begin
    dims = (2, 2, 2, 2)
    i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

    a = randn(elt, i, j)
    # TODO: Should this be allowed?
    # TODO: Add support for specifying new name.
    for f in
        (factorize, left_orth, left_polar, lq, orth, polar, qr, right_orth, right_polar)
      x, y = f(a, (i,))
      @test x * y ≈ a
    end

    a = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    for f in
        (factorize, left_orth, left_polar, lq, orth, polar, qr, right_orth, right_polar)
      x, y = f(a, (i, k), (j, l))
      @test x * y ≈ a
    end
  end
  @testset "svd" begin
    dims = (2, 2, 2, 2)
    i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

    a = randn(elt, i, j)
    # TODO: Should this be allowed?
    # TODO: Add support for specifying new name.
    u, s, v = svd(a, (i,))
    @test u * s * v ≈ a

    a = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    u, s, v = svd(a, (i, k), (j, l))
    @test u * s * v ≈ a

    # Test truncation.
    a = randn(elt, i, j, k, l)
    u, s, v = svd(a, (i, k), (j, l); trunc=(; maxrank=2))
    @test u * s * v ≉ a
    @test Int.(Tuple(size(s))) == (2, 2)
  end
  @testset "left_null/right_null" begin
    dims = (2, 2, 2, 2)
    i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

    a = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    for n in (left_null(a, (i, k), (j, l)), left_null(a, (i, k)))
      @test (i, k) ⊆ nameddimsindices(n)
      @test norm(n * a) ≈ 0
    end
    for n in (right_null(a, (i, k), (j, l)), right_null(a, (i, k)))
      @test (j, l) ⊆ nameddimsindices(n)
      @test norm(n * a) ≈ 0
    end
  end
end
