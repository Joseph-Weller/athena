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
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif

namespace {
Real gconst;
Real njeans;
Real m_refine;
}  // namespace

Real vector_potential(int component,
                      Real my_x1, Real my_x2, Real my_x3,
                      Real x1c, Real x2c, Real x3c,
                      Real I0, Real r0, Real rsurf, Real c, Real angle);

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
    m_refine = pin->GetReal("problem","m_refine");
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

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(1);
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
  Real Poly_cut = 1.0-atm_merge;

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
  if (MAGNETIC_FIELDS_ENABLED) {
    AthenaArray<Real> a1, a2, a3;
    Real a1_target, a1_impactor, a2_target, a2_impactor, a3_target, a3_impactor;
    int nx1 = block_size.nx1 + 2*NGHOST;
    int nx2 = block_size.nx2 + 2*NGHOST;
    int nx3 = block_size.nx3 + 2*NGHOST;
    a1.NewAthenaArray(nx3,nx2,nx1);
    a2.NewAthenaArray(nx3,nx2,nx1);
    a3.NewAthenaArray(nx3,nx2,nx1);

    Real x1c_target   = delx_1;
    Real x2c_target   = dely_1;
    Real x3c_target   = 0.0;

    Real x1c_impactor = delx_2;
    Real x2c_impactor = dely_2;
    Real x3c_impactor = 0.0;

    // Dipole parameters
    Real c              = pin->GetReal("problem","c");
    Real I0_target      = pin->GetReal("problem","I0_target");
    Real I0_impactor    = pin->GetReal("problem","I0_impactor");
    Real rsurf_target   = rad_max_1;
    Real rsurf_impactor = rad_max_2;
    Real angle_target   = pin->GetReal("problem","angle_target")*PI/180.;
    Real angle_impactor = pin->GetReal("problem","angle_impactor")*PI/180.;
    Real r0_target      = rad_max_1/3.0;
    Real r0_impactor    = rad_max_2/3.0;

    int level = loc.level;
    // Initialize vector potential
    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
    for (int i=is; i<=ie+1; ++i) {
      if ((pbval->nblevel[1][0][1]>level && j==js)
          || (pbval->nblevel[1][2][1]>level && j==je+1)
          || (pbval->nblevel[0][1][1]>level && k==ks)
          || (pbval->nblevel[2][1][1]>level && k==ke+1)
          || (pbval->nblevel[0][0][1]>level && j==js   && k==ks)
          || (pbval->nblevel[0][2][1]>level && j==je+1 && k==ks)
          || (pbval->nblevel[2][0][1]>level && j==js   && k==ke+1)
          || (pbval->nblevel[2][2][1]>level && j==je+1 && k==ke+1)) {

        Real x1l = pcoord->x1f(i)+0.25*pcoord->dx1f(i);
        Real x1r = pcoord->x1f(i)+0.75*pcoord->dx1f(i);

        a1_target   = 0.5*(vector_potential(X1DIR,
                                            x1l, pcoord->x2f(j), pcoord->x3f(k),
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target) +
                           vector_potential(X1DIR,
                                            x1r, pcoord->x2f(j), pcoord->x3f(k),
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target));

        a1_impactor   = 0.5*(vector_potential(X1DIR,
                                              x1l, pcoord->x2f(j), pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_potential(X1DIR,
                                              x1r, pcoord->x2f(j), pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));
      } else {
        a1_target   = vector_potential(X1DIR,
                                       pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                       x1c_target, x2c_target, x3c_target,
                                       I0_target, r0_target,
                                       rsurf_target, c, angle_target);

        a1_impactor = vector_potential(X1DIR,
                                       pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                       x1c_impactor, x2c_impactor, x3c_impactor,
                                       I0_impactor, r0_impactor,
                                       rsurf_impactor, c, angle_impactor);
      }

      if ((pbval->nblevel[1][1][0]>level && i==is)
          || (pbval->nblevel[1][1][2]>level && i==ie+1)
          || (pbval->nblevel[0][1][1]>level && k==ks)
          || (pbval->nblevel[2][1][1]>level && k==ke+1)
          || (pbval->nblevel[0][1][0]>level && i==is   && k==ks)
          || (pbval->nblevel[0][1][2]>level && i==ie+1 && k==ks)
          || (pbval->nblevel[2][1][0]>level && i==is   && k==ke+1)
          || (pbval->nblevel[2][1][2]>level && i==ie+1 && k==ke+1)) {

        Real x2l = pcoord->x2f(j)+0.25*pcoord->dx2f(j);
        Real x2r = pcoord->x2f(j)+0.75*pcoord->dx2f(j);

        a2_target   = 0.5*(vector_potential(X2DIR,
                                            pcoord->x1f(i), x2l, pcoord->x3f(k),
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target) +
                           vector_potential(X2DIR,
                                            pcoord->x1f(i), x2r, pcoord->x3f(k),
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target));

        a2_impactor   = 0.5*(vector_potential(X2DIR,
                                              pcoord->x1f(i), x2l, pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_potential(X2DIR,
                                              pcoord->x1f(i), x2r, pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));

      } else {
        a2_target   = vector_potential(X2DIR,
                                       pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                       x1c_target, x2c_target, x3c_target,
                                       I0_target, r0_target,
                                       rsurf_target, c, angle_target);

        a2_impactor = vector_potential(X2DIR,
                                       pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                       x1c_impactor, x2c_impactor, x3c_impactor,
                                       I0_impactor, r0_impactor,
                                       rsurf_impactor, c, angle_impactor);
      }

      if ((pbval->nblevel[1][1][0]>level && i==is)
          || (pbval->nblevel[1][1][2]>level && i==ie+1)
          || (pbval->nblevel[1][0][1]>level && j==js)
          || (pbval->nblevel[1][2][1]>level && j==je+1)
          || (pbval->nblevel[1][0][0]>level && i==is   && j==js)
          || (pbval->nblevel[1][0][2]>level && i==ie+1 && j==js)
          || (pbval->nblevel[1][2][0]>level && i==is   && j==je+1)
          || (pbval->nblevel[1][2][2]>level && i==ie+1 && j==je+1)) {

        Real x3l = pcoord->x3f(k)+0.25*pcoord->dx3f(k);
        Real x3r = pcoord->x3f(k)+0.75*pcoord->dx3f(k);

        a3_target   = 0.5*(vector_potential(X3DIR,
                                            pcoord->x1f(i), pcoord->x2f(j), x3l,
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target) +
                           vector_potential(X3DIR,
                                            pcoord->x1f(i), pcoord->x2f(j), x3r,
                                            x1c_target, x2c_target, x3c_target,
                                            I0_target, r0_target,
                                            rsurf_target, c, angle_target));

        a3_impactor   = 0.5*(vector_potential(X3DIR,
                                              pcoord->x1f(i), pcoord->x2f(j), x3l,
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_potential(X3DIR,
                                              pcoord->x1f(i), pcoord->x2f(j), x3r,
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));
      } else {
        a3_target   = vector_potential(X3DIR,
                                       pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                       x1c_target, x2c_target, x3c_target,
                                       I0_target, r0_target,
                                       rsurf_target, c, angle_target);

        a3_impactor = vector_potential(X3DIR,
                                       pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                       x1c_impactor, x2c_impactor, x3c_impactor,
                                       I0_impactor, r0_impactor,
                                       rsurf_impactor, c, angle_impactor);
      }

      a1(k,j,i) = a1_target + a1_impactor;
      a2(k,j,i) = a2_target + a2_impactor;
      a3(k,j,i) = a3_target + a3_impactor;
    }}}

    // initialize interface B
    Real isqrtfourpi = 1.0/std::sqrt(4.0*PI);
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = isqrtfourpi *
                                 ((a3(k  ,j+1,i) - a3(k,j,i))/pcoord->dx2f(j) -
                                  (a2(k+1,j  ,i) - a2(k,j,i))/pcoord->dx3f(k));
        }
      }
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x2f(k,j,i) = isqrtfourpi *
                                 ((a1(k+1,j,i  ) - a1(k,j,i))/pcoord->dx3f(k) -
                                  (a3(k  ,j,i+1) - a3(k,j,i))/pcoord->dx1f(i));
        }
      }
    }

    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = isqrtfourpi *
                                 ((a2(k,j  ,i+1) - a2(k,j,i))/pcoord->dx1f(i) -
                                  (a1(k,j+1,i  ) - a1(k,j,i))/pcoord->dx2f(j));
        }
      }
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) += 0.5*
                            (SQR(0.5*(pfield->b.x1f(k,j,i)+pfield->b.x1f(k  ,j  ,i+1)))
                            +SQR(0.5*(pfield->b.x2f(k,j,i)+pfield->b.x2f(k  ,j+1,i  )))
                            +SQR(0.5*(pfield->b.x3f(k,j,i)+pfield->b.x3f(k+1,j  ,i  ))));
        }
      }
    }

    a1.DeleteAthenaArray();
    a2.DeleteAthenaArray();
    a3.DeleteAthenaArray();
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m, vol;
  FaceField &b = pfield->b;

  face1.NewAthenaArray ((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);
  vol.NewAthenaArray   ((ie-is)+2*NGHOST+1);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face1Area (k,   j,   is, ie+1, face1);
      pcoord->Face2Area (k,   j+1, is, ie,   face2p);
      pcoord->Face2Area (k,   j,   is, ie,   face2m);
      pcoord->Face3Area (k+1, j,   is, ie,   face3p);
      pcoord->Face3Area (k,   j,   is, ie,   face3m);
      pcoord->CellVolume(k,   j,   is, ie,   vol);
      for (int i=is; i<=ie; ++i) {
        user_out_var(0,k,j,i) = (face1 (i+1)*b.x1f(k,j,i+1)-face1 (i)*b.x1f(k,j,i) +
                                 face2p(i  )*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i) +
                                 face3p(i  )*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
        user_out_var(0,k,j,i) /= vol(i);
      }
    }
  }
}

// computes Cartesian components of rotated dipole vector potential A_\phi
Real vector_potential(int component,
                      Real my_x1, Real my_x2, Real my_x3,
                      Real x1c, Real x2c, Real x3c,
                      Real I0, Real r0, Real rsurf, Real c, Real angle) {

  // rotation
  Real x_rot = my_x1-x1c;
  Real y_rot = (my_x2-x2c)*std::cos(angle) + (my_x3-x3c)*std::sin(angle);
  Real z_rot = (my_x3-x3c)*std::cos(angle) - (my_x2-x2c)*std::sin(angle);

  // intermediate quantities
  Real r_rot_sq = std::pow(x_rot, 2.) + std::pow(y_rot, 2.) + std::pow(z_rot, 2.);
  Real wbar_sq  = std::pow(x_rot, 2.) + std::pow(y_rot, 2.);
  Real r0_sq    = std::pow(r0, 2.);

  // evaluate three components of Cartesian vector potential
  Real a_val = (I0*PI*r0_sq*(1. + (15.*r0_sq*(r0_sq + wbar_sq)) /
                (8.*std::pow(r0_sq + r_rot_sq, 2.))))/(c*std::pow(r0_sq + r_rot_sq, 1.5));
  Real ax_rot = -1. * y_rot * a_val;
  Real ay_rot = x_rot * a_val;
  Real az_rot = 0;
  if (component == X1DIR) {
    return ax_rot;
  } else if (component == X2DIR) {
    return (ay_rot*std::cos(angle) - az_rot*std::sin(angle));
  } else { // component == X3DIR
    return (ay_rot*std::sin(angle) + az_rot*std::cos(angle));
  }
}


// AMR refinement condition
int JeansCondition(MeshBlock *pmb) {
  Real njmin = 1e300;
  Real mass  = 0.0;
  const Real dx = pmb->pcoord->dx1f(0);  // assuming uniform cubic cells
  const Real vol = dx*dx*dx;
  const Real gamma = pmb->peos->GetGamma();
  const Real fac = 2.0*PI*std::sqrt(gamma)/dx;
  for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
    for (int j=pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
      for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
	      // Real dxi = pmb->pcoord->dx1f(i);
	      // Real vol = dxi*dxi*dxi
        Real nj = fac*std::sqrt(pmb->phydro->w(IPR,k,j,i))/pmb->phydro->w(IDN,k,j,i);
        njmin = std::min(njmin, nj);
	      Real m_amount = vol*pmb->phydro->u(IDN,k,j,i);
	      mass = std::max(mass, m_amount);
      }
    }
  }
  if (mass > m_refine)
    return 1;
  if (njmin < njeans)
    return 1;
  if (mass < m_refine * 0.1)
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
