```python
using LinearAlgebra
using Statistics

using GomalizingFlow
using GomalizingFlow: PhysicalParams, HyperParams
using MLUtils
```

```python
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example3d.toml")
hp = GomalizingFlow.load_hyperparams(configpath);
pp = hp.pp
```

```python
module My

using GomalizingFlow: PhysicalParams

struct HMC{T <: AbstractFloat, N}
    cfgs::Array{T,N} # configurations
    p::Array{T,N} # momentum
    cfgs_old::Array{T,N} # configurations
    p_old::Array{T,N} # momentum
    F::Array{T,N} # Force
end

function HMC{T}(pp::PhysicalParams; init::Function=zeros) where T<:AbstractFloat    
    lattice_shape = pp.lattice_shape
    HMC{T, length(lattice_shape)}(
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
    )
end

HMC(pp::PhysicalParams; kwargs...) = HMC{Float64}(pp::PhysicalParams; kwargs...)

Base.eltype(::HMC{T, N}) where {T, N} = T

end # module

using .My
```

```python
function calc_potential(Φ::My.HMC{T, N}, pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    m²=T(pp.m²)
    λ =T(pp.λ)
    ϕ = Φ.cfgs
    p = Φ.p
    V = zero(T)
    for iz=1:Lz
        for iy=1:Ly
            for ix=1:Lx
                V += m² * ϕ[ix,iy,iz]^2+ λ * ϕ[ix,iy,iz]^4
            end
        end
    end
    return V
end
```

```python
function calc_kinetic(Φ::My.HMC{T, N},pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    m²=T(pp.m²)
    λ =T(pp.λ)
    ϕ = Φ.cfgs
    K = zero(T)
    for iz=1:Lz
        for iy=1:Ly
            for ix=1:Lx
                ip=ix+1
                im=ix-1
                if ip>Lx #  1<= ip <= L 
                    ip-=Lx
                end
                if im<1
                    im+=Lx
                end
                K-=ϕ[ix,iy,iz]*(ϕ[ip,iy,iz]+ϕ[im,iy,iz]-2ϕ[ix,iy,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iy+1
                im=iy-1
                if ip>Ly
                    ip-=Ly
                end
                if im<1
                    im+=Ly
                end
                K-=ϕ[ix,iy,iz]*(ϕ[ix,ip,iz]+ϕ[ix,im,iz]-2ϕ[ix,iy,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iz+1
                im=iz-1
                if ip>Lz
                    ip-=Lz
                end
                if im<1
                    im+=Lz
                end
                K-=ϕ[ix,iy,iz]*(ϕ[ix,iy,ip]+ϕ[ix,iy,im]-2ϕ[ix,iy,iz])
            end
        end
    end
    return K
end
```

```python
Φ = My.HMC(pp, init=rand);
```

```python
action = GomalizingFlow.ScalarPhi4Action(pp.m², pp.λ)
@time action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
```

```python
@assert calc_potential(Φ, pp) ≈ sum(action.m² * Φ.cfgs .^2 + action.λ * Φ.cfgs .^ 4)
@assert calc_kinetic(Φ, pp) + calc_potential(Φ, pp) ≈ action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
```

```python
hamiltonian = dot(Φ.p, Φ.p) + action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
```

```python
function calc_force!(Φ::My.HMC,pp::PhysicalParams)
    Lx, Ly, Lz = pp.lattice_shape
    m²=pp.m²
    λ =pp.λ
    F = Φ.F
    ϕ = Φ.cfgs
    
    for iz=1:Lz
        for iy=1:Ly
            for ix=1:Lx
                F[ix,iy,iz] = -m²*ϕ[ix,iy,iz] -λ*ϕ[ix,iy,iz]^3/12
                # = = = = = = =
                ixp=ix+1
                ixm=ix-1
                if ixp>Lx
                    ixp-=Lx
                end
                if ixm<1
                    ixm+=Lx
                end
                F[ix,iy,iz]+= ϕ[ixp,iy,iz]+ϕ[ixm,iy,iz]-2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                iyp=iy+1
                iym=iy-1
                if iyp>Ly
                    iyp-=Ly
                end
                if iym<1
                    iym+=Ly
                end
                F[ix,iy,iz]+= ϕ[ix,iyp,iz]+ϕ[ix,iym,iz]-2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                izp=iz+1
                izm=iz-1
                if izp>Lz
                    izp-=Lz
                end
                if izm<1
                    izm+=Lz
                end
                F[ix,iy,iz]+= ϕ[ix,iy,izp]+ϕ[ix,iy,izm]-2ϕ[ix,iy,iz]
            end
        end
    end
end
```

```python
function metropolis!(Φ::My.HMC, pp::PhysicalParams)
    lattice_shape = pp.lattice_shape
    Φ.cfgs_old .= copy(Φ.cfgs)
    Φ.p .= rand(lattice_shape...)
    Φ.p_old .= copy(Φ.p)
    
    S_old = action(unsqueeze(Φ.cfgs_old, dims=ndims(Φ.cfgs_old)+1))[begin]
    Σp_old² = dot(Φ.p_old, Φ.p_old) # faster than sum(Φ.p_old * Φ.p_old)
    H_old = 0.5Σp_old² + S_old # Hamiltonian
    
    Nmd = 200
    ϵ = inv(Nmd)
    for _ in 1:Nmd
        #Φ.p .+= ϵ/2 * Φ.F
        Φ.cfgs .+= ϵ * Φ.p
        #Φ.p .+= ϵ/2 * Φ.F
    end
    
    S = action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
    Σp² = dot(Φ.p, Φ.p) # faster than sum(Φ.p * Φ.p)
    H = 0.5Σp² + S # Hamiltonian
    
    ΔH = H - H_old
    ξ = rand(eltype(Φ))
    if ξ < exp(-ΔH)
        return true, ΔH
    else
        # restore cfgs from old cfgs
        Φ.cfgs .= copy(Φ.cfgs_old)
        return false, ΔH
    end
end 
```

```python
function calcgreen(Φ::My.HMC,pp::PhysicalParams)
    example_loc = CartesianIndex(repeat([1], ndims(Φ.cfgs))...)
    volume = prod(pp.lattice_shape)
    
    #             sum          vec
    # (Lx, Ly, Lz) -> (1,1,Lz) -> (Lz)
    return sum(
        Φ.cfgs[example_loc] * Φ.cfgs,
        dims=1:(ndims(Φ.cfgs)-1)
    )/volume |> vec
end
```

```python
function calc_force!(Φ::HMCFieldSet,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    m²=parameters.m²
    λ =parameters.λ
    F = Φ.F
    ϕ = Φ.ϕ
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                F[ix,iy,iz] = -m²*ϕ[ix,iy,iz] -λ*ϕ[ix,iy,iz]^3/12
                # = = = = = = =
                ixp=ix+1
                ixm=ix-1
                if ixp>Lx
                    ixp-=Lx
                end
                if ixm<1
                    ixm+=Lx
                end
                F[ix,iy,iz]+= ϕ[ixp,iy,iz]+ϕ[ixm,iy,iz]-2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                iyp=iy+1
                iym=iy-1
                if iyp>Ly
                    iyp-=Ly
                end
                if iym<1
                    iym+=Ly
                end
                F[ix,iy,iz]+= ϕ[ix,iyp,iz]+ϕ[ix,iym,iz]-2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                izp=iz+1
                izm=iz-1
                if izp>Lz
                    izp-=Lz
                end
                if izm<1
                    izm+=Lz
                end
                F[ix,iy,iz]+= ϕ[ix,iy,izp]+ϕ[ix,iy,izm]-2ϕ[ix,iy,iz]
            end
        end
    end
end
```

```python
function runHMC(hp::HyperParams, ntrials=20)
    T = Float64
    pp::PhysicalParams = hp.pp
    N = length(pp.lattice_shape)
    action = GomalizingFlow.ScalarPhi4Action{T}(pp.m², pp.λ)
    Φ = My.HMC{T}(pp, init=rand)
    history = (cond=T[], ΔH=T[], accepted=Bool[], Green=Vector{T}[])
    for i in 1:ntrials
        cond = mean(Φ.cfgs)
        accepted, ΔH = metropolis!(Φ, pp)
        push!(history[:cond], cond)
        push!(history[:ΔH], ΔH)
        push!(history[:accepted], accepted)
        push!(history[:Green], calcgreen(Φ, pp))
    end
    return history
end
```

```python
runHMC(hp)
```
