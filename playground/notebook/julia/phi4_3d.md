# HMC for phi4
- A. Tomiya June/13/2021
- A. Tomiya Jan/20/2022 parameter search


# $\phi^4$ theory

In ths note, we implement lattice $\phi4$ theory on the lattice.


```julia
using Random
using Plots
using StatsBase
using Statistics
using Printf
```

```julia
struct Coordinate
    Lx::Int
    Ly::Int
    Lz::Int
    V::Int
end
function Coordinate(Lx,Ly,Lz)
    V = Lx*Ly*Lz
    return Coordinate(Lx,Ly,Lz,V)
end;
```

```julia
struct Parameters
    coordinate::Coordinate
    m²::Float64
    λ::Float64
    Nmd::Int
    ϵ::Float64
end
function Parameters(Lx,Ly,Lz,m²,λ,Nmd::Int)
    ϵ=1/Nmd
    coordinate=Coordinate(Lx,Ly,Lz)
    return Parameters(coordinate,m²,λ,Nmd,ϵ)
end
```

```julia
import Base
function Base.zeros(coordinate::Coordinate)
    Lx=coordinate.Lx
    Ly=coordinate.Ly
    Lz=coordinate.Lz
    return zeros(Lx,Ly,Lz)
end
function ScalarField(parameters::Parameters)
    coordinate=parameters.coordinate
    return zeros(coordinate)
end
```

```julia
struct HMCFieldSet
    ϕ::Array{Float64,3}
    p::Array{Float64,3}
    ϕold::Array{Float64,3}
    pold::Array{Float64,3}
    F::Array{Float64,3}
end
function HMCFieldSet(parameters::Parameters)
    ϕ = ScalarField(parameters);
    p = ScalarField(parameters);
    ϕold = ScalarField(parameters); # for metropolis test
    pold = ScalarField(parameters); # for metropolis test
    F = ScalarField(parameters); # force
    return HMCFieldSet(ϕ,p,ϕold,pold,F)
end
```

```julia
function calc_potential(Φ,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    m²=parameters.m²
    λ =parameters.λ
    ϕ = Φ.ϕ
    p = Φ.p
    V = 0.0
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                V+=ϕ[ix,iy,iz]^2*m²*0.5+ϕ[ix,iy,iz]^4*λ/24
                #V+=ϕ[ix,iy,iz]^2*m²+ϕ[ix,iy,iz]^4*λ
            end
        end
    end
    return V
end
#calc_potential(ϕ,parameters)
```

```julia
function calc_kinetic(Φ::HMCFieldSet,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    m²=parameters.m²
    λ =parameters.λ
    ϕ = Φ.ϕ
    K = 0.0
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
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
    return K/2
    #return K
end
# calc_kinetic(ϕ,parameters)
```

```julia
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
                F[ix,iy,iz] = -m²*ϕ[ix,iy,iz] -λ*ϕ[ix,iy,iz]^3/6
                #F[ix,iy,iz] = -2m²*ϕ[ix,iy,iz] -4λ*ϕ[ix,iy,iz]^3
                # = = = = = = =
                ixp=ix+1
                ixm=ix-1
                if ixp>Lx
                    ixp-=Lx
                end
                if ixm<1
                    ixm+=Lx
                end
                C = 4
                D = 2
                
                F[ix,iy,iz]+= (ϕ[ixp,iy,iz]+ϕ[ixm,iy,iz]-C*ϕ[ix,iy,iz])/D
                # - - - - - - - - - - - - - - - - - - - -
                iyp=iy+1
                iym=iy-1
                if iyp>Ly
                    iyp-=Ly
                end
                if iym<1
                    iym+=Ly
                end
                F[ix,iy,iz]+= (ϕ[ix,iyp,iz]+ϕ[ix,iym,iz]-C*ϕ[ix,iy,iz])/D
                # - - - - - - - - - - - - - - - - - - - -
                izp=iz+1
                izm=iz-1
                if izp>Lz
                    izp-=Lz
                end
                if izm<1
                    izm+=Lz
                end
                F[ix,iy,iz]+= (ϕ[ix,iy,izp]+ϕ[ix,iy,izm]-C*ϕ[ix,iy,iz])/D
            end
        end
    end
end
```

```julia
function update_p!(Φ::HMCFieldSet,ϵ,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    ϕ = Φ.ϕ
    p = Φ.p
    calc_force!(Φ,parameters)
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                p[ix,iy,iz] += ϵ*Φ.F[ix,iy,iz]
            end
        end
    end
end
```

```julia
function update_phi!(Φ::HMCFieldSet,ϵ,parameters::Parameters)
    ϕ = Φ.ϕ
    p = Φ.p
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                ϕ[ix,iy,iz]+=ϵ*p[ix,iy,iz]
            end
        end
    end
end
#p = randn(10,10,10)
#ϵ = 1
#update_phi!(ϕ,p,ϵ,parameters)
```

```julia
function MD!(Φ::HMCFieldSet,parameters::Parameters)# PQP integrator
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    ϵ = parameters.ϵ
    Nmd = parameters.Nmd
    for imd=1:Nmd
        update_p!(Φ, ϵ/2,  parameters)
        update_phi!(Φ, ϵ,  parameters)
        update_p!(Φ, ϵ/2,  parameters)
    end
end
```

```julia
function calc_action(Φ::HMCFieldSet,parameters::Parameters)
    K = calc_kinetic(Φ,parameters)
    V = calc_potential(Φ,parameters)
    return K+V
end
function calc_hamiltonian(Φ::HMCFieldSet,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    ϕ = Φ.ϕ
    p = Φ.p
    # summing up all p^2
    p2 = 0.0
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                p2+=p[ix,iy,iz]^2
            end
        end
    end
    S=calc_action(Φ,parameters)
    return p2/2+S
end
function calc_cond(Φ::HMCFieldSet,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    V =parameters.coordinate.V
    ϕ = Φ.ϕ
    cond=0.0
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                cond+=ϕ[ix,iy,iz]
            end
        end
    end
    return cond/V
end
```

```julia
function calc_2pt(Φ::HMCFieldSet,parameters::Parameters)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    V =parameters.coordinate.V
    ϕ = Φ.ϕ
    #println("Lz = $Lz")
    G = zeros(Float64, Lz)
    for iz=1:Lz
        for iy=1:Ly
            @simd for ix=1:Lx
                G[iz]+=ϕ[ix,iy,iz]*ϕ[1,1,1]
            end
        end
    end
    return G/V
end
```

```julia
function HMCupdater!(Φ::HMCFieldSet,parameters::Parameters,Metropolis=true)
    Lx=parameters.coordinate.Lx
    Ly=parameters.coordinate.Ly
    Lz=parameters.coordinate.Lz
    ϕold = Φ.ϕold # binding (copy pointer) 
    pold = Φ.pold # binding (copy pointer) 
    ϕ = Φ.ϕ # binding (copy pointer) 
    p = Φ.p # binding (copy pointer)
    # = = = = = = = = = = = = = 
    p .= randn(Lx,Ly,Lz) # refleshiing moomntum
    ϕold .= copy(ϕ) # copy value Met-test
    pold .= copy(p) # copy value
    Ho = calc_hamiltonian(Φ,parameters)
    MD!(Φ,parameters)
    #if 1==1 # reversibility test
    #    p .= -copy(p)
    #    MD!(Φ,parameters)
    #end
    if Metropolis
        Hn = calc_hamiltonian(Φ,parameters)
        dH = Hn - Ho
        ξ = rand(Float64) # 0<= ξ < 1
        #if dH<0
        #    return true,dH
        #elseif ξ<exp(-dH)
        #    return true,dH
        if ξ<exp(-dH) # dH<0 is included
            return true,dH
        else
            Φ.ϕ .= copy(ϕold) 
            #Φ.p .= copy(pold) 
            #Ho2 = calc_hamiltonian(Φ,parameters)
            #println("debug rejected Hold=$Ho =?= H=$(Ho2)  =!= $Hn, $ξ >? $(exp(-dH))")
            #println("dh=",dH)
            return false,dH
        end
    end
end
```

```julia
function run_hmc(m²,λ;Nmd = 10, Ntrj=20,LxLyLz=[4,4,4])
    Lx,Ly,Lz=LxLyLz[1],LxLyLz[2],LxLyLz[3]
    #println("Classical value = $( sqrt(6*abs(m²)/λ) ) ")
    parameters = Parameters(Lx,Ly,Lz,m²,λ,Nmd)
    Φ = HMCFieldSet(parameters);
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    hist_cond = zeros(Ntrj)
    flags = zeros(Ntrj)
    dHs = zeros(Ntrj)
    Gs = []
    for i = 1:Ntrj
        #S = calc_action(Φ,parameters)
        cond = calc_cond(Φ,parameters)
        flag, dH = HMCupdater!(Φ,parameters)
        hist_cond[i]=cond
        flags[i] = ifelse(flag,1,0)
        dHs[i]=dH
        push!(Gs,calc_2pt(Φ,parameters) )
        #=
        if i%100==0
            #println("$i $(S) $(cond) $flag")
            @printf("%5d %4f %4f %d %2f\n",i,S,cond,flag,mean(flags[1:i]) )
        end
        =#
    end
    # println("acceptance rate : $(mean(flags))")
    # plot(hist_cond.^2,xlabel="MC time", ylabel="<cond^2>")
    return hist_cond, flags, dHs, Gs
end 
```

```julia
#=
function autocorr(x)
    acf = crosscor(x, x) # , mode='full')
    idx = div(length(acf),2)
    return acf[(idx+1):end]
end
function calc_τ_ac(x)
    ρ = autocorr(x)
    idx = length(ρ[ρ .< 0])
    if idx < 1
        idx = length(ρ)
    end
    return sum(ρ[1:idx])+1/2
end;
=#
```

```julia
# https://arxiv.org/abs/hep-lat/0409106 
function autocorr_core(a,t,av)
    N=length(a)
    acf=0.0
    for i=1:N-t
       acf+=(a[i]-av)*(a[i+t]-av) 
    end
    return acf/(N-t)
end
function autocorr(x)
    N=length(x)
    av=mean(x)
    Γ=zeros(N)
    for t=0:N-1
       Γ[t+1]=autocorr_core(x,t,av) 
    end
    return Γ/Γ[1]
end
function find_first_zero(ρ)
    idx = 0
    for t=1:length(ρ)
        if ρ[t]<0
            idx=round(Int,t*1.2)
            break
        end
    end
    return idx
end
function calc_τ_ac(x)
    ρ = autocorr(x)
    idx = find_first_zero(ρ)
    return sum(ρ[1:idx])+1/2
end
#=
ρ = autocorr(hist_cond[Ndisc:Ntrj])
println(calc_τ_ac(hist_cond[Ndisc:Ntrj]))
plot(ρ[1:idx])
=#
```

```julia
function autocorr_error(ρ)
    Λ = 1000
    δρ2 = []
    N = length(ρ)
    for t=1:idx
        for k=1:t+Λ
            k2=k-t
            if k2<1
                k2+=length(ρ)
            end
            push!(δρ2, (ρ[k+t]+ρ[k2] - 2ρ[k]*ρ[t] )^2  )
        end
    end
    δρ2/=N
    return δρ2
end
function calc_τ_ac_we(x)
    ρ = autocorr(x)
    N = length(ρ)
    idx = find_first_zero(ρ)
    δρ2=autocorr_error(ρ)[1:idx]
    arr = ρ[1:idx] .< sqrt.(δρ2[1:idx]) 
    W=0
    for i=1:length(arr)
        if arr[i]>0
            W = i
            break
        end
    end
    τ=calc_τ_ac(x)
    δτ=τ*sqrt((2*W+2)/N)
    return τ,δτ
end
```

```julia
function half_2pt(Gav)
    Nz = length(Gav)
    Nz2=div(Nz,2)-1
    Gav2 = zeros(Nz2)
    Gav2[1]=Gav[1]
    for z = 2:Nz2
        Gav2[z] = (Gav[z] + Gav[Nz-z+1])/2
    end
    return Gav2
end
function calc_meff_core(Gav2)
    Nz2=length(Gav2)
    meffs=[]
    for t in 2:Nz2-1
        meff = -log(Gav2[t+1]/Gav2[t])
        push!(meffs,meff)
    end
    return meffs
end
function calc_meff(Gav)
    Gav2 = half_2pt(Gav)
    return calc_meff_core(Gav2)
end
```

# test HMC

```julia
# test HMC
# in the symmetric phase (massless)
#
# physical parameters
using Random
Random.seed!(54321)
m²= 0.0 # -0.4
λ = 1 
L = 12; LxLyLz = [L,L,L]
# HMC parameters
Nmd = 10
#Ntrj = 10^4*2
Ntrj = 10^4 * 2
Ndisc = div(Ntrj,20)
# - - -
hist_cond, flags, dHs, Gs = run_hmc(m²,λ,Nmd = Nmd, Ntrj=Ntrj,LxLyLz=LxLyLz)
println("acc=$(mean(flags)), cond=$(mean(hist_cond[Ndisc:Ntrj])), 1=$(mean(exp.(-dHs)))?")
plot(hist_cond,label=nothing,ylabel="cond",xlabel="trj")
```

```julia
goma
```

```julia
println("τ = $(calc_τ_ac(hist_cond[Ndisc:Ntrj]))")
ρ=autocorr(hist_cond[Ndisc:Ntrj])
idx = find_first_zero(ρ)
plot(collect(1:idx) .-1 ,ρ[1:idx],ylabel="τint",label="symmetric (m=0, λ>0)",ylim=(-.1,1.1))
```

```julia
println("τ = $(calc_τ_ac(hist_cond[Ndisc:Ntrj]))")
ρ=autocorr(hist_cond[Ndisc:Ntrj])
idx = find_first_zero(ρ)
δρ=sqrt.( autocorr_error(ρ)[1:idx] )
plot(collect(1:idx) .-1 ,ρ[1:idx],ribbon=δρ,ylabel="τint",label="symmetric (m=0, λ>0)",ylim=(-.1,1.1))
```

```julia
#Gav2=half_2pt(Gav)
#plot(Gav2,label=nothing)
```

```julia
τ = calc_τ_ac_we(hist_cond[Ndisc:Ntrj])
```

```julia
p=plot()
Lz = length(Gs[1])
tt = collect(1:Lz) .-1
Gav=[]
for z=1:Lz
    G=0
    for iconf=Ndisc:Ntrj
        G += Gs[iconf][z]
    end
    G/=length(Ndisc:Ntrj)
    push!(Gav,G)
end
scatter!(p, tt, Gav,label=nothing)
p
```

```julia
meffs = calc_meff(Gav)
println("m = $(sqrt(abs(m²))) ")
println("meff = $(mean(meffs)), mL=$(mean(meffs)*LxLyLz[3])")
plot(meffs,label=nothing)
```

- - - -

```julia
# test HMC
# in the symmetric phase (free)
#
# physical parameters
m²= 0.25
λ = 0
L = 12; LxLyLz = [L,L,L]
# HMC parameters
Nmd = 10
Ntrj = 10^5
Ndisc = div(Ntrj,20)
# - - -
hist_cond, flags, dHs, Gs = run_hmc(m²,λ,Nmd = Nmd, Ntrj=Ntrj,LxLyLz=LxLyLz)
println("acc=$(mean(flags)), cond=$(mean(hist_cond[Ndisc:Ntrj])), 1=$(mean(exp.(-dHs)))?")
plot(hist_cond,label=nothing,ylabel="cond",xlabel="trj")
```

```julia
println("τ = $(calc_τ_ac(hist_cond[Ndisc:Ntrj]))")
ρ=autocorr(hist_cond[Ndisc:Ntrj])
idx = find_first_zero(ρ)
δρ=sqrt.( autocorr_error(ρ)[1:idx] )
plot(collect(1:idx) .-1 ,ρ[1:idx],ribbon=δρ,ylabel="τint",label="symmetric (free)",ylim=(-.1,1.1))
```

```julia
τ = calc_τ_ac_we(hist_cond[Ndisc:Ntrj])
```

```julia
p=plot()
Lz = length(Gs[1])
tt = collect(1:Lz) .-1
Gav=[]
for z=1:Lz
    G=0
    for iconf=Ndisc:Ntrj
        G += Gs[iconf][z]
    end
    G/=length(Ndisc:Ntrj)
    push!(Gav,G)
end
scatter!(p, tt, Gav,label=nothing)
p
```

```julia
meffs = calc_meff(Gav)
println("m = $(sqrt(abs(m²))) ")
println("meff = $(mean(meffs)) $(std(meffs)/sqrt(length(meffs))) # mL=$(mean(meffs)*LxLyLz[3])")
plot(meffs,label=nothing)
```

- - - 

```julia
# test HMC
# broken phase (critical)
# physical parameters
m²= -0.4
λ = 5.113 # from E3 in 1904.12072 
L = 6; LxLyLz = [L,L,L*2]
# HMC parameters
Nmd = 200
Ntrj = 10^4*2
Ndisc = div(Ntrj,20)
# - - -
hist_cond, flags, dHs, Gs =  run_hmc(m²,λ,Nmd = Nmd, Ntrj=Ntrj,LxLyLz=LxLyLz)
println("acc=$(mean(flags)), cond=$(mean(hist_cond[Ndisc:Ntrj])), 1=$(mean(exp.(-dHs)))?")
plot((hist_cond),label=nothing,ylabel="cond",xlabel="trj",alpha=0.5,color="red")
plot!(abs.(hist_cond),label="abs",alpha=0.5,linestyle=:dash,color="black")
```

```julia
τ = calc_τ_ac(hist_cond[Ndisc:Ntrj])
println("τ = $(τ)")
ρ=autocorr(hist_cond[Ndisc:Ntrj])
idx = find_first_zero(ρ)
plot(collect(1:idx) .-1 ,ρ[1:idx],ylabel=raw"$\tau_{{\rm int}}$",label="broken (critical)",ylim=(-0.1,1.1))
```

```julia
τ = calc_τ_ac(hist_cond[Ndisc:Ntrj])
println("τ = $(τ)")
ρ=autocorr(hist_cond[Ndisc:Ntrj])
idx = find_first_zero(ρ)
δρ=sqrt.( autocorr_error(ρ)[1:idx] )
plot(collect(1:idx) .-1 ,ρ[1:idx],ribbon=δρ,ylabel=raw"$\tau_{{\rm int}}$",label="broken (critical)",ylim=(-0.1,1.1))
```

```julia
τ = calc_τ_ac_we(hist_cond[Ndisc:Ntrj])
```

```julia
p=plot()
Lz = length(Gs[1])
tt = collect(1:Lz) .-1
Gav=[]
for z=1:Lz
    G=0
    for iconf=Ndisc:Ntrj
        G += Gs[iconf][z]
    end
    G/=length(Ndisc:Ntrj)
    push!(Gav,G)
end
scatter!(p, tt, Gav,label=nothing)
p
```

```julia
meffs = calc_meff(Gav)
println("m = $(sqrt(abs(m²))) ")
println("meff = $(mean(meffs)), Lmeff=$(mean(meffs)*LxLyLz[3])")
plot(meffs,label=nothing)
```

## debug note (HMC)

<ul>
<li><input type="checkbox" checked="">rerversiblity</li>
<li><input type="checkbox" checked="">Hamiltonian check for rejected pattern to confirm configs are kept</li>
    <li><input type="checkbox" checked=""> av[e^{-dH}] = 1</li>
</ul>


- - - 
