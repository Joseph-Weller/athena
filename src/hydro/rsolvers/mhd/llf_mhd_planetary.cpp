//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file llf_mhd.cpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for MHD
//!
//! Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//! This flux is very diffusive, even more diffusive than HLLE, and so it is not
//! recommended for use in applications.  However, it is useful for testing, or for
//! problems where other Riemann solvers fail.  This special *_planetary version of the
//! LLF Riemann solver assumes a change from the primitive variable pressure to
//! specific internal energy for ease in interfacing with planetary EOS.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#include "../../hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The LLF Riemann solver for MHD (both adiabatic and isothermal)

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, const AthenaArray<Real> &bx,
                          AthenaArray<Real> &wl, AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &ey, AthenaArray<Real> &ez,
                          AthenaArray<Real> &wct, const AthenaArray<Real> &dxw) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[(NWAVE)],wri[(NWAVE)],du[(NWAVE)];
  Real flxi[(NWAVE)],fl[(NWAVE)],fr[(NWAVE)];

  Real dt = pmy_block->pmy_mesh->dt;

#pragma omp simd private(wli,wri,du,fl,fr,flxi)
  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables

    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    wli[IEN]=wl(IEN,i);
    wli[IBY]=wl(IBY,i);
    wli[IBZ]=wl(IBZ,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    wri[IEN]=wr(IEN,i);
    wri[IBY]=wr(IBY,i);
    wri[IBZ]=wr(IBZ,i);

    Real bxi = bx(k,j,i);

    Real pgas_l = pmy_block->peos->PresFromRhoEs(wli[IDN], wli[IEN]);
    Real pgas_r = pmy_block->peos->PresFromRhoEs(wri[IDN], wri[IEN]);

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real cfl = pmy_block->peos->FastMagnetosonicSpeed(wli,bxi);
    Real cfr = pmy_block->peos->FastMagnetosonicSpeed(wri,bxi);
    Real a = 0.5*std::max( (std::abs(wli[IVX]) + cfl), (std::abs(wri[IVX]) + cfr) );

    //--- Step 3.  Compute L/R fluxes

    Real mxl = wli[IDN]*wli[IVX];
    Real mxr = wri[IDN]*wri[IVX];
    Real pbl = 0.5*(bxi*bxi + SQR(wli[IBY]) + SQR(wli[IBZ]));
    Real pbr = 0.5*(bxi*bxi + SQR(wri[IBY]) + SQR(wri[IBZ]));

    fl[IDN] = mxl;
    fr[IDN] = mxr;

    fl[IVX] = mxl*wli[IVX] + pbl - SQR(bxi);
    fr[IVX] = mxr*wri[IVX] + pbr - SQR(bxi);

    fl[IVY] = mxl*wli[IVY] - bxi*wli[IBY];
    fr[IVY] = mxr*wri[IVY] - bxi*wri[IBY];

    fl[IVZ] = mxl*wli[IVZ] - bxi*wli[IBZ];
    fr[IVZ] = mxr*wri[IVZ] - bxi*wri[IBZ];

    Real el = (wli[IDN]*wli[IEN]
               + 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ])) + pbl);
    Real er = (wri[IDN]*wri[IEN]
               + 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ])) + pbr);
    fl[IVX] += pgas_l;
    fr[IVX] += pgas_r;
    fl[IEN] = (el + pgas_l + pbl - bxi*bxi)*wli[IVX];
    fr[IEN] = (er + pgas_r + pbr - bxi*bxi)*wri[IVX];
    fl[IEN] -= bxi*(wli[IBY]*wli[IVY] + wli[IBZ]*wli[IVZ]);
    fr[IEN] -= bxi*(wri[IBY]*wri[IVY] + wri[IBZ]*wri[IVZ]);

    fl[IBY] = wli[IBY]*wli[IVX] - bxi*wli[IVY];
    fr[IBY] = wri[IBY]*wri[IVX] - bxi*wri[IVY];

    fl[IBZ] = wli[IBZ]*wli[IVX] - bxi*wli[IVZ];
    fr[IBZ] = wri[IBZ]*wri[IVX] - bxi*wri[IVZ];

    //--- Step 4.  Compute difference in L/R states dU

    du[IDN] = wri[IDN]          - wli[IDN];
    du[IVX] = wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX];
    du[IVY] = wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY];
    du[IVZ] = wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ];
    du[IEN] = er - el;
    du[IBY] = wri[IBY] - wli[IBY];
    du[IBZ] = wri[IBZ] - wli[IBZ];

    //--- Step 5.  Compute the LLF flux at interface (see Toro eq. 10.42).

    flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];
    flxi[IBY] = 0.5*(fl[IBY] + fr[IBY]) - a*du[IBY];
    flxi[IBZ] = 0.5*(fl[IBZ] + fr[IBZ]) - a*du[IBZ];

    //--- Step 6. Store results into 3D array of fluxes

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    flx(IEN,k,j,i) = flxi[IEN];
    ey(k,j,i) = -flxi[IBY];
    ez(k,j,i) =  flxi[IBZ];

    wct(k,j,i)=GetWeightForCT(flxi[IDN], wli[IDN], wri[IDN], dxw(i), dt);
  }

  return;
}
