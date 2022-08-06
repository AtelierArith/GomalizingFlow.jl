{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ac23eb45-4d14-4fde-af2b-df6d9e10d79a",
   "metadata": {},
   "outputs": [],
   "source": [
    "using LinearAlgebra\n",
    "using Statistics\n",
    "\n",
    "using GomalizingFlow\n",
    "using GomalizingFlow: PhysicalParams, HyperParams\n",
    "using MLUtils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c19be30b-4456-4ab4-b6b4-93a44c27e2b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "┌ Info: Set device: CPU\n",
      "└ @ GomalizingFlow /Users/terasakisatoshi/work/atelier_arith/GomalizingFlow.jl/src/parameters.jl:14\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "PhysicalParams\n",
       "  L: Int64 8\n",
       "  Nd: Int64 3\n",
       "  M2: Float64 -4.0\n",
       "  lam: Float64 5.113\n"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "configpath = joinpath(pkgdir(GomalizingFlow), \"cfgs\", \"example3d.toml\")\n",
    "hp = GomalizingFlow.load_hyperparams(configpath);\n",
    "pp = hp.pp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e723ac77-78bd-420f-9abd-c56aa876ea6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "module My\n",
    "\n",
    "using GomalizingFlow: PhysicalParams\n",
    "\n",
    "struct HMC{T <: AbstractFloat, N}\n",
    "    cfgs::Array{T,N} # configurations\n",
    "    p::Array{T,N} # momentum\n",
    "    cfgs_old::Array{T,N} # configurations\n",
    "    p_old::Array{T,N} # momentum\n",
    "    F::Array{T,N} # Force\n",
    "end\n",
    "\n",
    "function HMC{T}(pp::PhysicalParams; init::Function=zeros) where T<:AbstractFloat    \n",
    "    lattice_shape = pp.lattice_shape\n",
    "    HMC{T, length(lattice_shape)}(\n",
    "        init(lattice_shape...),\n",
    "        init(lattice_shape...),\n",
    "        init(lattice_shape...),\n",
    "        init(lattice_shape...),\n",
    "        init(lattice_shape...),\n",
    "    )\n",
    "end\n",
    "\n",
    "HMC(pp::PhysicalParams; kwargs...) = HMC{Float64}(pp::PhysicalParams; kwargs...)\n",
    "\n",
    "Base.eltype(::HMC{T, N}) where {T, N} = T\n",
    "\n",
    "end # module\n",
    "\n",
    "using .My"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "79f631e1-92bc-4ed8-8b74-dde64cca2143",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "calc_potential (generic function with 1 method)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function calc_potential(Φ::My.HMC{T, N}, pp::PhysicalParams) where {T, N}\n",
    "    Lx, Ly, Lz = pp.lattice_shape\n",
    "    m²=T(pp.m²)\n",
    "    λ =T(pp.λ)\n",
    "    ϕ = Φ.cfgs\n",
    "    p = Φ.p\n",
    "    V = zero(T)\n",
    "    for iz=1:Lz\n",
    "        for iy=1:Ly\n",
    "            for ix=1:Lx\n",
    "                V += m² * ϕ[ix,iy,iz]^2+ λ * ϕ[ix,iy,iz]^4\n",
    "            end\n",
    "        end\n",
    "    end\n",
    "    return V\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "052bd458-7f28-4e77-9c1c-1e2274c8ccc9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "calc_kinetic (generic function with 1 method)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function calc_kinetic(Φ::My.HMC{T, N},pp::PhysicalParams) where {T, N}\n",
    "    Lx, Ly, Lz = pp.lattice_shape\n",
    "    m²=T(pp.m²)\n",
    "    λ =T(pp.λ)\n",
    "    ϕ = Φ.cfgs\n",
    "    K = zero(T)\n",
    "    for iz=1:Lz\n",
    "        for iy=1:Ly\n",
    "            for ix=1:Lx\n",
    "                ip=ix+1\n",
    "                im=ix-1\n",
    "                if ip>Lx #  1<= ip <= L \n",
    "                    ip-=Lx\n",
    "                end\n",
    "                if im<1\n",
    "                    im+=Lx\n",
    "                end\n",
    "                K-=ϕ[ix,iy,iz]*(ϕ[ip,iy,iz]+ϕ[im,iy,iz]-2ϕ[ix,iy,iz])\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                ip=iy+1\n",
    "                im=iy-1\n",
    "                if ip>Ly\n",
    "                    ip-=Ly\n",
    "                end\n",
    "                if im<1\n",
    "                    im+=Ly\n",
    "                end\n",
    "                K-=ϕ[ix,iy,iz]*(ϕ[ix,ip,iz]+ϕ[ix,im,iz]-2ϕ[ix,iy,iz])\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                ip=iz+1\n",
    "                im=iz-1\n",
    "                if ip>Lz\n",
    "                    ip-=Lz\n",
    "                end\n",
    "                if im<1\n",
    "                    im+=Lz\n",
    "                end\n",
    "                K-=ϕ[ix,iy,iz]*(ϕ[ix,iy,ip]+ϕ[ix,iy,im]-2ϕ[ix,iy,iz])\n",
    "            end\n",
    "        end\n",
    "    end\n",
    "    return K\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "879dfc67-c7c5-4b5a-baeb-1b2cd0f61d76",
   "metadata": {},
   "outputs": [],
   "source": [
    "Φ = My.HMC(pp, init=rand);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "303e7b54-c8ab-442b-91c6-0f2d724fd70e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  1.556262 seconds (5.60 M allocations: 296.452 MiB, 5.11% gc time, 99.93% compilation time)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "92.66439604352388"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action = GomalizingFlow.ScalarPhi4Action(pp.m², pp.λ)\n",
    "@time action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d8e54ace-cfac-4408-8e5f-a0af0f2c335c",
   "metadata": {},
   "outputs": [],
   "source": [
    "@assert calc_potential(Φ, pp) ≈ sum(action.m² * Φ.cfgs .^2 + action.λ * Φ.cfgs .^ 4)\n",
    "@assert calc_kinetic(Φ, pp) + calc_potential(Φ, pp) ≈ action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5f005936-8a85-497f-902c-9ff03ba929b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "270.6181497392381"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hamiltonian = dot(Φ.p, Φ.p) + action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "be9fc92d-abd9-40db-9493-47f060ac8813",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "calc_force! (generic function with 1 method)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function calc_force!(Φ::My.HMC,pp::PhysicalParams)\n",
    "    Lx, Ly, Lz = pp.lattice_shape\n",
    "    m²=pp.m²\n",
    "    λ =pp.λ\n",
    "    F = Φ.F\n",
    "    ϕ = Φ.cfgs\n",
    "    \n",
    "    for iz=1:Lz\n",
    "        for iy=1:Ly\n",
    "            for ix=1:Lx\n",
    "                F[ix,iy,iz] = -m²*ϕ[ix,iy,iz] -λ*ϕ[ix,iy,iz]^3/12\n",
    "                # = = = = = = =\n",
    "                ixp=ix+1\n",
    "                ixm=ix-1\n",
    "                if ixp>Lx\n",
    "                    ixp-=Lx\n",
    "                end\n",
    "                if ixm<1\n",
    "                    ixm+=Lx\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ixp,iy,iz]+ϕ[ixm,iy,iz]-2ϕ[ix,iy,iz]\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                iyp=iy+1\n",
    "                iym=iy-1\n",
    "                if iyp>Ly\n",
    "                    iyp-=Ly\n",
    "                end\n",
    "                if iym<1\n",
    "                    iym+=Ly\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ix,iyp,iz]+ϕ[ix,iym,iz]-2ϕ[ix,iy,iz]\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                izp=iz+1\n",
    "                izm=iz-1\n",
    "                if izp>Lz\n",
    "                    izp-=Lz\n",
    "                end\n",
    "                if izm<1\n",
    "                    izm+=Lz\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ix,iy,izp]+ϕ[ix,iy,izm]-2ϕ[ix,iy,iz]\n",
    "            end\n",
    "        end\n",
    "    end\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "63fc8d11-09c7-4226-ab01-852028cea011",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "metropolis! (generic function with 1 method)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function metropolis!(Φ::My.HMC, pp::PhysicalParams)\n",
    "    lattice_shape = pp.lattice_shape\n",
    "    Φ.cfgs_old .= copy(Φ.cfgs)\n",
    "    Φ.p .= rand(lattice_shape...)\n",
    "    Φ.p_old .= copy(Φ.p)\n",
    "    \n",
    "    S_old = action(unsqueeze(Φ.cfgs_old, dims=ndims(Φ.cfgs_old)+1))[begin]\n",
    "    Σp_old² = dot(Φ.p_old, Φ.p_old) # faster than sum(Φ.p_old * Φ.p_old)\n",
    "    H_old = 0.5Σp_old² + S_old # Hamiltonian\n",
    "    \n",
    "    Nmd = 200\n",
    "    ϵ = inv(Nmd)\n",
    "    for _ in 1:Nmd\n",
    "        #Φ.p .+= ϵ/2 * Φ.F\n",
    "        Φ.cfgs .+= ϵ * Φ.p\n",
    "        #Φ.p .+= ϵ/2 * Φ.F\n",
    "    end\n",
    "    \n",
    "    S = action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]\n",
    "    Σp² = dot(Φ.p, Φ.p) # faster than sum(Φ.p * Φ.p)\n",
    "    H = 0.5Σp² + S # Hamiltonian\n",
    "    \n",
    "    ΔH = H - H_old\n",
    "    ξ = rand(eltype(Φ))\n",
    "    if ξ < exp(-ΔH)\n",
    "        return true, ΔH\n",
    "    else\n",
    "        # restore cfgs from old cfgs\n",
    "        Φ.cfgs .= copy(Φ.cfgs_old)\n",
    "        return false, ΔH\n",
    "    end\n",
    "end "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8184f6eb-47dd-450a-9956-45a8aaa912d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "calcgreen (generic function with 1 method)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function calcgreen(Φ::My.HMC,pp::PhysicalParams)\n",
    "    example_loc = CartesianIndex(repeat([1], ndims(Φ.cfgs))...)\n",
    "    volume = prod(pp.lattice_shape)\n",
    "    \n",
    "    #             sum          vec\n",
    "    # (Lx, Ly, Lz) -> (1,1,Lz) -> (Lz)\n",
    "    return sum(\n",
    "        Φ.cfgs[example_loc] * Φ.cfgs,\n",
    "        dims=1:(ndims(Φ.cfgs)-1)\n",
    "    )/volume |> vec\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "702d2311-07c5-4206-a1f7-8e43680b2c2b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "calc_force! (generic function with 1 method)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function calc_force!(Φ::HMCFieldSet,parameters::Parameters)\n",
    "    Lx=parameters.coordinate.Lx\n",
    "    Ly=parameters.coordinate.Ly\n",
    "    Lz=parameters.coordinate.Lz\n",
    "    m²=parameters.m²\n",
    "    λ =parameters.λ\n",
    "    F = Φ.F\n",
    "    ϕ = Φ.ϕ\n",
    "    for iz=1:Lz\n",
    "        for iy=1:Ly\n",
    "            @simd for ix=1:Lx\n",
    "                F[ix,iy,iz] = -m²*ϕ[ix,iy,iz] -λ*ϕ[ix,iy,iz]^3/12\n",
    "                # = = = = = = =\n",
    "                ixp=ix+1\n",
    "                ixm=ix-1\n",
    "                if ixp>Lx\n",
    "                    ixp-=Lx\n",
    "                end\n",
    "                if ixm<1\n",
    "                    ixm+=Lx\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ixp,iy,iz]+ϕ[ixm,iy,iz]-2ϕ[ix,iy,iz]\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                iyp=iy+1\n",
    "                iym=iy-1\n",
    "                if iyp>Ly\n",
    "                    iyp-=Ly\n",
    "                end\n",
    "                if iym<1\n",
    "                    iym+=Ly\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ix,iyp,iz]+ϕ[ix,iym,iz]-2ϕ[ix,iy,iz]\n",
    "                # - - - - - - - - - - - - - - - - - - - -\n",
    "                izp=iz+1\n",
    "                izm=iz-1\n",
    "                if izp>Lz\n",
    "                    izp-=Lz\n",
    "                end\n",
    "                if izm<1\n",
    "                    izm+=Lz\n",
    "                end\n",
    "                F[ix,iy,iz]+= ϕ[ix,iy,izp]+ϕ[ix,iy,izm]-2ϕ[ix,iy,iz]\n",
    "            end\n",
    "        end\n",
    "    end\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a25e3aa8-af67-4b78-bb44-ff902c4cfa03",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "runHMC (generic function with 2 methods)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function runHMC(hp::HyperParams, ntrials=20)\n",
    "    T = Float64\n",
    "    pp::PhysicalParams = hp.pp\n",
    "    N = length(pp.lattice_shape)\n",
    "    action = GomalizingFlow.ScalarPhi4Action{T}(pp.m², pp.λ)\n",
    "    Φ = My.HMC{T}(pp, init=rand)\n",
    "    history = (cond=T[], ΔH=T[], accepted=Bool[], Green=Vector{T}[])\n",
    "    for i in 1:ntrials\n",
    "        cond = mean(Φ.cfgs)\n",
    "        accepted, ΔH = metropolis!(Φ, pp)\n",
    "        push!(history[:cond], cond)\n",
    "        push!(history[:ΔH], ΔH)\n",
    "        push!(history[:accepted], accepted)\n",
    "        push!(history[:Green], calcgreen(Φ, pp))\n",
    "    end\n",
    "    return history\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "89d4a8c6-7965-403c-9eb7-de96f24c6d39",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(cond = [0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882, 0.4931469581324882], ΔH = [3615.6842826904663, 3251.172088917882, 3640.0622469305213, 3405.8607701756227, 3609.9323408284513, 3065.2539189835666, 3149.400533362185, 3508.8108789279418, 3352.1598035619563, 3230.790778548177, 3593.3686233214994, 3083.612250346456, 3567.711937705489, 3336.1782516959274, 3394.573825838215, 3149.4002788346647, 3767.4335330723775, 3139.9711636314287, 3155.125888582464, 3416.631451954773], accepted = Bool[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Green = [[0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771], [0.008095774909060072, 0.006124204823036271, 0.008237783795587013, 0.006919923571307094, 0.008311643648621822, 0.007666354339664201, 0.008313968369794617, 0.00831800180736771]])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "runHMC(hp)"
   ]
  }
 ],
 "metadata": {
  "jupytext": {
   "cell_metadata_filter": "-all",
   "notebook_metadata_filter": "-all"
  },
  "kernelspec": {
   "display_name": "Julia-nthread-16 1.7.3",
   "language": "julia",
   "name": "julia-nthread-16-1.7"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
