using PyCall

pushfirst!(PyVector(pyimport("sys")."path"), "")

@testset "ScalarPhi4Action" begin
	pyaction = pyimport("pymod.action")
	cfgs = pyaction.cfgs

	M2 = 1.0
	lam = 1.0
	out1 = LFT.ScalarPhi4Action(M2, lam)(cfgs |> LFT.reversedims)
	pyout1 = pyaction.out1
	@test out1 ≈ pyout1

	M2 = -4.0
	lam = 8.0
	out2 = LFT.ScalarPhi4Action(M2, lam)(cfgs |> LFT.reversedims)
	pyout2 = pyaction.out2
	@test out2 ≈ pyout2
end

