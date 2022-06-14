//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file giant_impact.cpp
//! \brief Problem generator for an idealized planetary giant impact
//!

// C headers

// C++ headers
#include <algorithm>
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif

namespace {
Real gconst;
Real njeans;
}  // namespace

int JeansCondition(MeshBlock *pmb);
void NoInflowInnerX1(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void NoInflowInnerX2(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void NoInflowOuterX2(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void NoInflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void NoInflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gconst = pin->GetOrAddReal("problem", "grav_const", 1.0);
  SetGravitationalConstant(gconst);
  SetGravityThreshold(0.0);  // NOTE(@pdmullen): as far as I know, not used in FMG
  if (adaptive) {
    njeans = pin->GetReal("problem","njeans");
    EnrollUserRefinementCondition(JeansCondition);
  }

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, NoInflowInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, NoInflowOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, NoInflowInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, NoInflowOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, NoInflowInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, NoInflowOuterX3);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief giant impact problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // gas constant
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;

  // ambient background
  Real da = pin->GetOrAddReal("problem", "damb", 1.0);
  Real pa = pin->GetOrAddReal("problem", "pamb", 1.0);

  // compute masses of the colliding bodies
  Real mtot = pin->GetOrAddReal("problem", "mtot", 1.1);
  Real mrat = pin->GetOrAddReal("problem", "mrat", 0.091);
  Real mass_2 = mrat*mtot;
  Real mass_1 = mtot - mass_2;

  // compute polytrope centers as offset from coordinate origin
  Real x_disp = pin->GetReal("problem", "x_disp");
  Real y_disp = pin->GetReal("problem", "y_disp");
  Real delx_1 = -1.0*(mass_2/mtot)*x_disp;
  Real delx_2 = (mass_1/mtot)*x_disp;
  Real dely_1 = -1.0*(mass_2/mtot)*y_disp;
  Real dely_2 = (mass_1/mtot)*y_disp;

  // velocities for each colliding body
  Real vcoll = pin->GetOrAddReal("problem", "vcoll", 0.0);
  Real delvx_1 = -1.0*(mass_2/mtot)*vcoll;
  Real delvx_2 = (mass_1/mtot)*vcoll;

  // compute central densities and pressures of planetary bodies
  // uses Earth and Mars benchmarks to scale polytropes and calculate radii
  // applies a linear fitting in mass to find central density, scales radius to have
  // appropriate mass.
  // NOTE(@pdmullen): what is this doing, @Joseph-Weller?
  Real dcrat_1 = pin->GetOrAddReal("problem", "dc_earth", 1.0);
  Real dcrat_2 = pin->GetOrAddReal("problem", "dc_mars", 1.0);
  Real dc_1 = ((dcrat_1 - dcrat_2)/(1.0 - 0.107))*(mass_1 - 0.107)+dcrat_2;
  Real dc_2 = ((dcrat_1 - dcrat_2)/(1.0 - 0.107))*(mass_2-  0.107)+dcrat_2;
  Real rad_max_1 = std::cbrt((PI/4.0)*(mass_1/dc_1));
  Real rad_max_2 = std::cbrt((PI/4.0)*(mass_2/dc_2));
  Real pc_1 = (2.0*gconst/PI)*SQR(dc_1*rad_max_1);
  Real pc_2 = (2.0*gconst/PI)*SQR(dc_2*rad_max_2);
  
  //input parameters for atmoshere merger and extent
  Real atm_merge = pin->GetOrAddReal("problem", "atm_merge", 0.03);
  Real atm_ext = pin->GetOrAddReal("problem", "atm_ext", 0.3);
  Real Poly_cut = 1.0-atm_merge

  // define collision origin
  Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("problem", "z0", 0.0);

  // setup giant impact
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // extract coordinate positions
        Real x = pcoord->x1v(i);
        Real y = pcoord->x2v(j);
        Real z = pcoord->x3v(k);
        Real rad_1 = std::sqrt(SQR(x-(x0+delx_1)) + SQR(y-(y0+dely_1)) + SQR(z-z0));
        Real rad_2 = std::sqrt(SQR(x-(x0+delx_2)) + SQR(y-(y0+dely_2)) + SQR(z-z0));

        // TODO(@Joseph-Weller):
        // (1) some of the coefficients below seem rather arbitrary. Can we derive them in
        //     a more general way? Or at least make them an input parameter?
        // (2) let's take a really close look at this logic and make sure it is giving us
        //     exactly what we want.  For example, if we are inside the atmosphere of
        //     body 1 but also inside planetary body 2, do we want the density at that
        //     location to be the sum of those densities?
        Real den = da; Real pres = pa;
        Real momx = 0.0; Real momy = 0.0; Real momz = 0.0;
        if (rad_1 <= rad_max_1*atm_ext) {  // inside planetary body 1 body+atmosphere
          if (rad_1 <= rad_max_1*Poly_cut) {  // inside planetary body 1
            den = dc_1*rad_max_1/(PI*rad_1)*std::sin((PI*rad_1)/rad_max_1);
            pres = pc_1*std::pow(rad_max_1/(PI*rad_1)*std::sin((PI*rad_1)/rad_max_1),2.0);
          } else {  // inside planetary body 1 atmosphere
            den = (dc_1*1.0/(PI*Poly_cut) *
                   std::sin((PI*Poly_cut))*std::pow(rad_max_1*Poly_cut/rad_1,15.0));
            pres = (pc_1*std::pow(1.0/(PI*Poly_cut)*std::sin((PI*Poly_cut)),2.0) *
                    std::pow(rad_max_1*Poly_cut/rad_1,16.0));
          }
          momx = den*delvx_1;
        }

        if (rad_2 <= rad_max_2*atm_ext) {  // inside planetary body 2 body+atmosphere
          if (rad_2 <= rad_max_2*Poly_cut) {  // inside planetary body 2
            den = dc_2*rad_max_2/(PI*rad_2)*std::sin((PI*rad_2)/rad_max_2);
            pres = pc_2*std::pow(rad_max_2/(PI*rad_2)*std::sin((PI*rad_2)/rad_max_2),2.0);
          } else {  // inside planetary body 2 atmosphere
            den = (dc_2*1.0/(PI*Poly_cut) *
                   std::sin((PI*Poly_cut))*std::pow(rad_max_2*Poly_cut/rad_2,15.0));
            pres = (pc_2*std::pow(1.0/(PI*Poly_cut)*std::sin((PI*Poly_cut)),2.0) *
                    std::pow(rad_max_2*Poly_cut/rad_2,16.0));
          }
          momx = den*delvx_2;
        }

        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = momx;
        phydro->u(IM2,k,j,i) = momy;
        phydro->u(IM3,k,j,i) = momz;
        phydro->u(IEN,k,j,i) = pres/gm1;
        phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                     SQR(phydro->u(IM2,k,j,i)) +
                                     SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
      }
    }
  }
}

// AMR refinement condition
int JeansCondition(MeshBlock *pmb) {
  Real njmin = 1e300;
  const Real dx = pmb->pcoord->dx1f(0);  // assuming uniform cubic cells
  const Real gamma = pmb->peos->GetGamma();
  const Real fac = 2.0*PI*std::sqrt(gamma)/dx;
  for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
    for (int j=pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
      for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
        Real nj = fac*std::sqrt(pmb->phydro->w(IPR,k,j,i))/pmb->phydro->w(IDN,k,j,i);
        njmin = std::min(njmin, nj);
      }
    }
  }

  if (njmin < njeans)
    return 1;
  if (njmin > njeans * 2.5)
    return -1;
  return 0;
}

void NoInflowInnerX1(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVX,k,j,il) > 0.0) {
          prim(IVX,k,j,il-i) = 0.0;
        } else {
          prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
        }
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
      }
    }
  }
}

void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVX,k,j,iu) < 0.0) {
          prim(IVX,k,j,iu+i) = 0.0;
        } else {
          prim(IVX,k,j,iu+i) = prim(IVX,k,j,iu);
        }
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
      }
    }
  }
}

void NoInflowInnerX2(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,jl-j,i) = prim(IDN,k,jl,i);
        prim(IVX,k,jl-j,i) = prim(IVX,k,jl,i);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVY,k,jl,i) > 0.0) {
          prim(IVY,k,jl-j,i) = 0.0;
        } else {
          prim(IVY,k,jl-j,i) = prim(IVY,k,jl,i);
        }
        prim(IVZ,k,jl-j,i) = prim(IVZ,k,jl,i);
        prim(IPR,k,jl-j,i) = prim(IPR,k,jl,i);
      }
    }
  }
}

void NoInflowOuterX2(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,ju+j,i) = prim(IDN,k,ju,i);
        prim(IVX,k,ju+j,i) = prim(IVX,k,ju,i);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVY,k,ju,i) < 0.0) {
          prim(IVY,k,ju+j,i) = 0.0;
        } else {
          prim(IVY,k,ju+j,i) = prim(IVY,k,ju,i);
        }
        prim(IVZ,k,ju+j,i) = prim(IVZ,k,ju,i);
        prim(IPR,k,ju+j,i) = prim(IPR,k,ju,i);
      }
    }
  }
}

void NoInflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        prim(IDN,kl-k,j,i) = prim(IDN,kl,j,i);
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) > 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        prim(IPR,kl-k,j,i) = prim(IPR,kl,j,i);
      }
    }
  }
  return;
}

void NoInflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) < 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
      }
    }
  }
  return;
}
