//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bump.cpp
//! \brief Problem generator for advecting bump
//!
//! REFERENCE: me

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace {
Real gconst;
Real njeans;
Real m_refine;
}  // namespace

Real vector_pot(int component,
                Real my_x1, Real my_x2, Real my_x3,
                Real x1c, Real x2c, Real x3c,
                Real I0, Real r0, Real rsurf, Real c, Real angle);

int JeansCondition(MeshBlock *pmb);

Real Mag_En_R(MeshBlock *pmb, int iout);
Real Mag_En_phi(MeshBlock *pmb, int iout);
Real Mag_En_z(MeshBlock *pmb, int iout);

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

void EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func, const char *name,
                               UserHistoryOperation op=UserHistoryOperation::sum);

//void srcmask(AthenaArray<Real> &src, int is, int ie, int js, int je,
//             int ks, int ke, const MGCoordinates &coord);

//void srcmask(AthenaArray<Real> &src, int is, int ie, int js, int je,
//             int ks, int ke, const MGCoordinates &coord) {
//  constexpr Real maskr = 2.3e10;
//  for (int k=ks; k<=ke; ++k) {
//    Real z = coord.x3v(k);
//    for (int j=js; j<=je; ++j) {
//      Real y = coord.x2v(j);
//      for (int i=is; i<=ie; ++i) {
//        Real x = coord.x1v(i);
//        Real r = std::sqrt(x*x + y*y + z*z);
//        if (r > maskr)
///          src(k, j, i) = 0.0;
//      }
//    }
//  }
//  return;
//}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gconst = pin->GetOrAddReal("problem", "grav_const", 1.0);
  SetGravitationalConstant(gconst);
  //EnrollUserMGGravitySourceMaskFunction(srcmask);
  //SetGravityThreshold(0.0);  // NOTE(@pdmullen): as far as I know, not used in FMG
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

  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, Mag_En_R, "EBr");
  EnrollUserHistoryOutput(1, Mag_En_phi, "EBphi");
  EnrollUserHistoryOutput(2, Mag_En_z, "EBz");

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
    {
      AllocateUserOutputVariables(5);
      return;
    }

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  
  
  Real ea   = pin->GetOrAddReal("problem", "eamb", 1.0); 
  Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  Real xvel = pin->GetOrAddReal("problem", "vx", 0.0);
  Real yvel = pin->GetOrAddReal("problem", "vy", 0.0);
  
  Real R_max_E = pin->GetOrAddReal("problem", "r_max_e", 1.0);
  Real ecent_E = pin->GetOrAddReal("problem", "ecent_e", 1.0);
  Real dc_E = pin->GetOrAddReal("problem", "dcent_e", 1.0);

  Real den_stab_E[10000];
  Real espec_stab_E[10000];
  Real r_store_E[10000];
  Real r0_E = 0.01;
  Real cs_2_E = 0.0;
  
  Real den_curr_E = dc_E;
  Real espec_curr_E = ecent_E;
  Real dr_E = R_max_E/10000;
  Real drho_E = 0.0;
  Real despec_E = 0.0;
  Real dM_E = 0.0;
  Real M_e = 0.0;
  Real pres_E = peos->PresFromRhoEs(den_curr_E, espec_curr_E);


  Real R_max_I = pin->GetOrAddReal("problem", "r_max_i", 1.0);
  Real ecent_I = pin->GetOrAddReal("problem", "ecent_i", 1.0);
  Real dc_I = pin->GetOrAddReal("problem", "dcent_i", 1.0);

  Real den_stab_I[10000];
  Real espec_stab_I[10000];
  Real r_store_I[10000];
  Real r0_I = 0.01;
  Real cs_2_I = 0.0;

  Real den_curr_I = dc_I;
  Real espec_curr_I = ecent_I;
  Real dr_I = R_max_I/10000;
  Real drho_I = 0.0;
  Real despec_I = 0.0;
  Real dM_I = 0.0;
  Real M_I = 0.0;
  Real pres_I = peos->PresFromRhoEs(den_curr_I, espec_curr_I);

  //Real espec_max = 0.0;
  //Real en_max = 0.0;
  //while (pres >0){
  //std::cout << den_curr_E <<std::endl;
  
  int place = 0;
  while(r0_E<R_max_E){
    den_stab_E[place] = den_curr_E;
    espec_stab_E[place] = espec_curr_E;
    r_store_E[place]= r0_E;

    cs_2_E = peos->AsqFromRhoEs(den_curr_E, espec_curr_E);
    pres_E = peos->PresFromRhoEs(den_curr_E, espec_curr_E);
    drho_E = -1*gconst*M_e*den_curr_E/(cs_2_E*std::pow(r0_E, 2));
    despec_E = (pres_E/pow(den_curr_E, 2))*drho_E;
    dM_E = 4*PI*den_curr_E*pow(r0_E,2);

    den_curr_E = den_curr_E + dr_E*drho_E;
    espec_curr_E = espec_curr_E + dr_E*despec_E;
    M_e = M_e + dr_E*dM_E;
    r0_E = r0_E+dr_E;


    den_stab_I[place] = den_curr_I;
    espec_stab_I[place] = espec_curr_I;
    r_store_I[place]= r0_I;

    cs_2_I = peos->AsqFromRhoEs(den_curr_I, espec_curr_I);
    pres_I = peos->PresFromRhoEs(den_curr_I, espec_curr_I);
    drho_I = -1*gconst*M_I*den_curr_I/(cs_2_I*std::pow(r0_I, 2));
    despec_I = (pres_I/pow(den_curr_I, 2))*drho_I;
    dM_I = 4*PI*den_curr_I*pow(r0_I,2);

    den_curr_I = den_curr_I + dr_I*drho_I;
    espec_curr_I = espec_curr_I + dr_I*despec_I;
    M_I = M_I + dr_I*dM_I;
    r0_I = r0_I+dr_I;


    //std::cout << M_e <<std::endl;
    place = place +1;


  }
  //std::cout << M_e <<std::endl;
  //std::cout << espec_stab[0] << std::endl;
  //std::cout << espec_stab[5000] << std::endl;

  // compute masses of the colliding bodies
  Real mass_2 = M_I;
  Real mass_1 = M_e;
  Real mtot = M_I + M_e;
  //std::cout << mass_2 << std::endl;

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

  Real atm_merge = pin->GetOrAddReal("problem", "atm_merge", 0.03);
  Real atm_ext = pin->GetOrAddReal("problem", "atm_ext", 0.3);
  Real Poly_cut = 1.0-atm_merge;


  Real x1_0   = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0   = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0   = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }
  int position = 0; 
  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
	Real rad2;
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - (x0+delx_1)) + SQR(y-(y0+dely_1)) + SQR(z-z0-0.1));
	  rad2 = std::sqrt(SQR(x-(x0+delx_2)) + SQR(y-(y0+dely_2)) + SQR(z-z0));
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x-(x0+delx_1)) + SQR(y-(y0+dely_1)) + SQR(z-z0-0.1));
	  rad2 = std::sqrt(SQR(x-(x0+delx_2)) + SQR(y-(y0+dely_2)) + SQR(z-z0));
        } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x-(x0+delx_1)) + SQR(y-(y0+dely_1)) + SQR(z-z0-0.1));
	  rad2 = std::sqrt(SQR(x-(x0+delx_2)) + SQR(y-(y0+dely_2)) + SQR(z-z0));
        }

        Real den = da;
        Real den_spot_E = 0.0;
        Real espec = ea;
        Real espec_spot_E = 0.0;
        Real momx = 0.0;
        Real momy = 0.0;
        Real kin  = 0.0;
	int spot = 0;
	Real r_spot = 0.0;


	Real den_spot_I = 0.0;
	Real espec_spot_I = 0.0;
	//std::cout << position << std::endl;
	
	//Earth use rad, _E, delvx_1
	if (rad < R_max_E*atm_ext) {
          if (rad < R_max_E*Poly_cut) {
	    while (r_spot <rad){
	      r_spot = r_spot + dr_E;
	      spot = spot + 1;
	    }
	    //std::cout << "here if before error, otherwise it won't print" << std::endl;
	    //std::cout << "current rad" << rad <<"current rspot"<< r_spot<< std::endl;
	    if (rad < 0.01){
	      den_spot_E = dc_E;
              espec_spot_E = ecent_E;
	    } else{
	      den_spot_E = den_stab_E[(spot-1)]+(rad - r_store_E[(spot-1)])*(den_stab_E[spot]-den_stab_E[(spot-1)])/(r_store_E[spot]-r_store_E[(spot-1)]);
	      espec_spot_E = espec_stab_E[(spot-1)]+(rad - r_store_E[(spot-1)])*(espec_stab_E[spot]-espec_stab_E[(spot-1)])/(r_store_E[spot]-r_store_E[(spot-1)]);
	    //from stored value interpolate to point
	    }
	    //den_stab = 1;
	    //den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);
            //std::exp(1/(std::pow(rcrit, 2))-1/(std::pow((rad - rcrit), 2)));
            den = den_spot_E;
            espec = espec_spot_E+ea;
	  
            momx = den*delvx_1;
            momy = den*yvel;
            kin = 0.5*den*delvx_1*delvx_1+0.5*den*yvel*yvel;
          } else {
            while (r_spot < R_max_E*Poly_cut){
              r_spot = r_spot + dr_E;
              spot = spot + 1;
            } 
	    //std::cout << r_spot << std::endl;
	    den = den_stab_E[(spot-1)] * std::pow(R_max_E*Poly_cut/rad,15.0);

            espec = espec_stab_E[(spot-1)] * R_max_E*Poly_cut/rad;

	    momx = den*delvx_1;
            momy = den*yvel;
            kin = 0.5*den*delvx_1*delvx_1+0.5*den*yvel*yvel;
	  }
        
	}

        //Impactor use rad2, _I, delvx_2
        if (rad2 < R_max_I*atm_ext) {
          if (rad2 < R_max_I*Poly_cut) {
            while (r_spot <rad2){
              r_spot = r_spot + dr_I;
              spot = spot + 1;
            }
            //std::cout << "here if before error, otherwise it won't print" << std::endl;
            //std::cout << "current rad" << rad <<"current rspot"<< r_spot<< std::endl;
            if (rad2 < 0.01){
              den_spot_I = dc_I;
              espec_spot_I = ecent_I;
            } else{
              den_spot_I = den_stab_I[(spot-1)]+(rad2 - r_store_I[(spot-1)])*(den_stab_I[spot]-den_stab_I[(spot-1)])/(r_store_I[spot]-r_store_I[(spot-1)]);
              espec_spot_I = espec_stab_I[(spot-1)]+(rad2 - r_store_I[(spot-1)])*(espec_stab_I[spot]-espec_stab_I[(spot-1)])/(r_store_I[spot]-r_store_I[(spot-1)]);
            //from stored value interpolate to point
            }
            //den_stab = 1;
            //den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);
            //std::exp(1/(std::pow(rcrit, 2))-1/(std::pow((rad - rcrit), 2)));
            den = den_spot_I;
            espec = espec_spot_I+ea;

            momx = den*delvx_2;
            momy = den*yvel;
            kin = 0.5*den*delvx_2*delvx_2+0.5*den*yvel*yvel;
          } else {
            while (r_spot < R_max_I*Poly_cut){
              r_spot = r_spot + dr_I;
              spot = spot + 1;
            }
            //std::cout << r_spot << std::endl;
            den = den_stab_I[(spot-1)] * std::pow(R_max_I*Poly_cut/rad2,15.0);

            espec = espec_stab_I[(spot-1)] * R_max_I*Poly_cut/rad2;

            momx = den*delvx_2;
            momy = den*yvel;
            kin = 0.5*den*delvx_2*delvx_2+0.5*den*yvel*yvel;
          }

        }

	//testing if something is wrong with assignment
	//phydro->u(IDN,k,j,i) = da;
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = momx;
        phydro->u(IM2,k,j,i) = momy;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = den*espec + kin;
	//std::cout << phydro->u(IEN,k,j,i) << std::endl;
	//std::cout << den*espec << std::endl;
        if (PLANETARY_EOS){
	  //std::cout << "here" << std::endl;
	  //gets here successfully so this isn't issue
	  //phydro->u(IEN,k,j,i) = den*espec + kin;
        
	  //phydro->u(IEN,k,j,i) = den*ea + kin;
	  if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
            phydro->u(IEN,k,j,i) += den;
	}
      }
    }
  }
 //std::cout << espec_max << std::endl;
 //std::cout << en_max << std::endl;
 //std::cout << "here" << std::endl;
 //gets through initialization
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
    Real rsurf_target   = R_max_E;
    Real rsurf_impactor = R_max_I;
    Real angle_target   = pin->GetReal("problem","angle_target")*PI/180.;
    Real angle_impactor = pin->GetReal("problem","angle_impactor")*PI/180.;
    Real r0_target      = R_max_E/3.0;
    Real r0_impactor    = R_max_I/3.0;

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

	a1_target   = 0.5*(vector_pot(X1DIR,
                                      x1l, pcoord->x2f(j), pcoord->x3f(k),
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target) +
                           vector_pot(X1DIR,
                                      x1r, pcoord->x2f(j), pcoord->x3f(k),
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target));

        a1_impactor   = 0.5*(vector_pot(X1DIR,
                                              x1l, pcoord->x2f(j), pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_pot(X1DIR,
                                              x1r, pcoord->x2f(j), pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));
      } else {
        a1_target   = vector_pot(X1DIR,
                                 pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a1_impactor = vector_pot(X1DIR,
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

	a2_target   = 0.5*(vector_pot(X2DIR,
                                      pcoord->x1f(i), x2l, pcoord->x3f(k),
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target) +
                           vector_pot(X2DIR,
                                      pcoord->x1f(i), x2r, pcoord->x3f(k),
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target));

        a2_impactor   = 0.5*(vector_pot(X2DIR,
                                              pcoord->x1f(i), x2l, pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_pot(X2DIR,
                                              pcoord->x1f(i), x2r, pcoord->x3f(k),
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));
      } else {
        a2_target   = vector_pot(X2DIR,
                                 pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a2_impactor = vector_pot(X2DIR,
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

	a3_target   = 0.5*(vector_pot(X3DIR,
                                      pcoord->x1f(i), pcoord->x2f(j), x3l,
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target) +
                           vector_pot(X3DIR,
                                      pcoord->x1f(i), pcoord->x2f(j), x3r,
                                      x1c_target, x2c_target, x3c_target,
                                      I0_target, r0_target,
                                      rsurf_target, c, angle_target));

        a3_impactor   = 0.5*(vector_pot(X3DIR,
                                              pcoord->x1f(i), pcoord->x2f(j), x3l,
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor) +
                             vector_pot(X3DIR,
                                              pcoord->x1f(i), pcoord->x2f(j), x3r,
                                              x1c_impactor, x2c_impactor, x3c_impactor,
                                              I0_impactor, r0_impactor,
                                              rsurf_impactor, c, angle_impactor));

      } else {
        a3_target   = vector_pot(X3DIR,
                                 pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a3_impactor = vector_pot(X3DIR,
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
    }}}

    a1.DeleteAthenaArray();
    a2.DeleteAthenaArray();
    a3.DeleteAthenaArray();
  }
  constexpr int scalar_norm = NSCALARS > 0 ? NSCALARS : 1.0;
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            Real rad_1 = std::sqrt(SQR(x-(x0+delx_1)) + SQR(y-(y0+dely_1)) + SQR(z-z0));
            Real rad_2 = std::sqrt(SQR(x-(x0+delx_2)) + SQR(y-(y0+dely_2)) + SQR(z-z0));
            if (n < 1) {
              if (rad_1 <= R_max_E*atm_ext) {
                pscalars->s(n,k,j,i) = 1.0/scalar_norm*phydro->u(IDN,k,j,i);
                //pscalars->s(n,k,j,i) = 1.0/scalar_norm;
              } else {
                pscalars->s(n,k,j,i) = 0.0;
              }
            } else {
              if (rad_2 <= R_max_I*atm_ext) {
                pscalars->s(n,k,j,i) = 1.0/scalar_norm*phydro->u(IDN,k,j,i);
                //pscalars->s(n,k,j,i) = 1.0/scalar_norm;
              } else {
                pscalars->s(n,k,j,i) = 0.0;
              }
            }
          }
        }
      }
    }
  }
 //std::cout << "here" << std::endl;
 //gets here
}

Real Mag_En_R(MeshBlock *pmb, int iout)
{
  //std::cout << "here" << std::endl;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  FaceField &b = pmb->pfield->b;
  Real EBr = 0;
  //Real EBphi = 0;
  //Real EBz = 0;
  Real r_rho_x = 0.0;
  Real r_rho_y = 0.0;
  Real r_rho_z = 0.0;
  Real rho_max = 1e-30;
  Real rho_test = 1e-30;
  Real R_earth = 6.378e8; //cgs
  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST+1);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        rho_test = pmb->phydro->u(IDN,k,j,i);
        if (rho_max <= rho_test) {
          r_rho_x = pmb->pcoord->x1v(i);
          r_rho_y = pmb->pcoord->x2v(j);
          r_rho_z = pmb->pcoord->x3v(k);
        }
        rho_max = std::max(rho_test,rho_max);
      }
    }
  }

  Real r_max = 4*R_earth;
  Real r_min = 3*R_earth;
  Real z_mag = R_earth;
  

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k,   j,   is, ie,   vol);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real ang = 0.0;
        Real rad_rho = std::sqrt(SQR(x-r_rho_x) + SQR(y-r_rho_y));
        Real y_pos = y-r_rho_y;
        Real x_pos = x-r_rho_x;
        Real z_pos = std::abs(z-r_rho_z);
        if (x_pos == 0.0 && y_pos > 0.0){
          ang = PI/2.0;
        } else if (x_pos == 0.0 && y_pos < 0.0){
          ang = -1.0*PI/2.0;
        } else if (x_pos < 0.0){
          ang = PI + std::atan((y_pos/x_pos));
        } else if (x_pos == 0.0){
          ang = 0.0;
        } else {
          ang = std::atan((y_pos/x_pos));
        }
        Real inside = 1.0;
        if (rad_rho > r_max || rad_rho < r_min || z_pos > z_mag){
          inside = 0.0;
        }
        Real Br = std::cos(ang)*b.x1f(k,j,i) + std::sin(ang)*b.x2f(k,j,i);
        Real Bphi = -1.0*std::sin(ang)*b.x1f(k,j,i) + std::cos(ang)*b.x2f(k,j,i);
        Real Bz = b.x3f(k,j,i);

        //EBr
        EBr += inside*std::pow(Br,2.0)/(8.0*PI)*vol(i);
        //EBphi
        //EBphi += inside*std::pow(Bphi,2.0)/(8.0*PI);
        //EBz
        //EBz += inside*std::pow(Bz,2.0)/(8.0*PI);
      }
    }
  }
  return EBr;
}

Real Mag_En_phi(MeshBlock *pmb, int iout)
{
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  FaceField &b = pmb->pfield->b;
  //Real EBr = 0;
  Real EBphi = 0;
  //Real EBz = 0;
  Real r_rho_x = 0.0;
  Real r_rho_y = 0.0;
  Real r_rho_z = 0.0;
  Real rho_max = 1e-30;
  Real rho_test = 1e-30;
  Real R_earth = 6.378e8; //cgs
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        rho_test = pmb->phydro->u(IDN,k,j,i);
        if (rho_max <= rho_test) {
          r_rho_x = pmb->pcoord->x1v(i);
          r_rho_y = pmb->pcoord->x2v(j);
          r_rho_z = pmb->pcoord->x3v(k);
        }
        rho_max = std::max(rho_test,rho_max);
      }
    }
  }

  Real r_max = 4*R_earth;
  Real r_min = 3*R_earth;
  Real z_mag = R_earth;

  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST+1);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k,   j,   is, ie,   vol);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real ang = 0.0;
        Real rad_rho = std::sqrt(SQR(x-r_rho_x) + SQR(y-r_rho_y));
        Real y_pos = y-r_rho_y;
        Real x_pos = x-r_rho_x;
        Real z_pos = std::abs(z-r_rho_z);
        if (x_pos == 0.0 && y_pos > 0.0){
          ang = PI/2.0;
        } else if (x_pos == 0.0 && y_pos < 0.0){
          ang = -1.0*PI/2.0;
        } else if (x_pos < 0.0){
          ang = PI + std::atan((y_pos/x_pos));
        } else if (x_pos == 0.0){
          ang = 0.0;
        } else {
          ang = std::atan((y_pos/x_pos));
        }
        Real inside = 1.0;
        if (rad_rho > r_max || rad_rho < r_min || z_pos > z_mag){
          inside = 0.0;
        }
        Real Br = std::cos(ang)*b.x1f(k,j,i) + std::sin(ang)*b.x2f(k,j,i);
        Real Bphi = -1.0*std::sin(ang)*b.x1f(k,j,i) + std::cos(ang)*b.x2f(k,j,i);
        Real Bz = b.x3f(k,j,i);

        //EBr
        //EBr += inside*std::pow(Br,2.0)/(8.0*PI);
        //EBphi
        EBphi += inside*std::pow(Bphi,2.0)/(8.0*PI)*vol(i);
        //EBz
        //EBz += inside*std::pow(Bz,2.0)/(8.0*PI);
      }
    }
  }
  return EBphi;
}

Real Mag_En_z(MeshBlock *pmb, int iout)
{
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  FaceField &b = pmb->pfield->b;
  //Real EBr = 0;
  //Real EBphi = 0;
  Real EBz = 0;
  Real r_rho_x = 0.0;
  Real r_rho_y = 0.0;
  Real r_rho_z = 0.0;
  Real rho_max = 1e-30;
  Real rho_test = 1e-30;
  Real R_earth = 6.378e8; //cgs
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        rho_test = pmb->phydro->u(IDN,k,j,i);
        if (rho_max <= rho_test) {
          r_rho_x = pmb->pcoord->x1v(i);
          r_rho_y = pmb->pcoord->x2v(j);
          r_rho_z = pmb->pcoord->x3v(k);
        }
        rho_max = std::max(rho_test,rho_max);
      }
    }
  }

  Real r_max = 4*R_earth;
  Real r_min = 3*R_earth;
  Real z_mag = R_earth;

  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST+1);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k,   j,   is, ie,   vol);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real ang = 0.0;
        Real rad_rho = std::sqrt(SQR(x-r_rho_x) + SQR(y-r_rho_y));
        Real y_pos = y-r_rho_y;
        Real x_pos = x-r_rho_x;
        Real z_pos = std::abs(z-r_rho_z);
        if (x_pos == 0.0 && y_pos > 0.0){
          ang = PI/2.0;
        } else if (x_pos == 0.0 && y_pos < 0.0){
          ang = -1.0*PI/2.0;
        } else if (x_pos < 0.0){
          ang = PI + std::atan((y_pos/x_pos));
        } else if (x_pos == 0.0){
          ang = 0.0;
        } else {
          ang = std::atan((y_pos/x_pos));
        }
        Real inside = 1.0;
        if (rad_rho > r_max || rad_rho < r_min || z_pos > z_mag){
          inside = 0.0;
        }
        Real Br = std::cos(ang)*b.x1f(k,j,i) + std::sin(ang)*b.x2f(k,j,i);
        Real Bphi = -1.0*std::sin(ang)*b.x1f(k,j,i) + std::cos(ang)*b.x2f(k,j,i);
        Real Bz = b.x3f(k,j,i);

        //EBr
        //EBr += inside*std::pow(Br,2.0)/(8.0*PI);
        //EBphi
        //EBphi += inside*std::pow(Bphi,2.0)/(8.0*PI);
        //EBz
        EBz += inside*std::pow(Bz,2.0)/(8.0*PI)*vol(i);
      }
    }
  }
  return EBz;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  if (MAGNETIC_FIELDS_ENABLED) {
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m, vol;
  FaceField &b = pfield->b;
  Real r_rho_x = 0.0;
  Real r_rho_y = 0.0;
  Real r_rho_z = 0.0;
  Real rho_max = 1e-30;
  Real rho_test = 1e-30;
  Real R_earth = 6.378e8; //cgs
  //divB calc
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
        rho_test = phydro->u(IDN,k,j,i);
        if (rho_max <= rho_test) {
          r_rho_x = pcoord->x1v(i);
          r_rho_y = pcoord->x2v(j);
          r_rho_z = pcoord->x3v(k);
        }
        rho_max = std::max(rho_test,rho_max);
      }
    }
  }
  } else{
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
    user_out_var(0,k,j,i) = 0.0;
    }
    }
    }
  }
  AthenaArray<Real> area, len2, len, len_p1;
  //z vorticity calc
  area.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len2.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len_p1.NewAthenaArray ((ie-is)+2*NGHOST+2);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face3Area(k,j,is,ie,area);
      pcoord->Edge2Length(k,j,is,ie+1,len2);
      for (int i=is; i<=ie; ++i) {
        user_out_var(1,k,j,i) -= (1.0/area(i))*(len2(i+1)*(phydro->u(IM2,k,j,i+1)/phydro->w(IDN,k,j,i+1)) -
                                                                len2(i)*(phydro->u(IM2,k,j,i)/phydro->w(IDN,k,j,i)));
      }
      pcoord->Edge1Length(k,j  ,is,ie,len);
      pcoord->Edge1Length(k,j+1,is,ie,len_p1);
      for (int i=is; i<=ie; ++i) {
        user_out_var(1,k,j,i) += (1.0/area(i))*(len_p1(i)*(phydro->u(IM1,k,j+1,i)/phydro->w(IDN,k,j+1,i)) -
                                                                len(i)*(phydro->u(IM1,k,j,i)/phydro->w(IDN,k,j,i)));
      }
    }
  }
  //z curl g calc
  //
  Real gy_0 = 0.0;
  Real gy_1 = 0.0;
  Real dx2 =0.0;

  Real gx_0 = 0.0;
  Real gx_1 =0.0;
  Real dx1 = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face3Area(k,j,is,ie,area);
      pcoord->Edge2Length(k,j,is,ie+1,len2);
      for (int i=is; i<=ie; ++i) {
        dx2 = pcoord->dx2v(j);
	gy_0 = 0.25*(pgrav->phi(k,j+1,i)-pgrav->phi(k,j-1,i)+pgrav->phi(k,j+1,i-1)-pgrav->phi(k,j-1,i-1))/dx2;
	gy_1 = 0.25*(pgrav->phi(k,j+1,i)-pgrav->phi(k,j-1,i)+pgrav->phi(k,j+1,i+1)-pgrav->phi(k,j-1,i+1))/dx2;
        user_out_var(2,k,j,i) -= (1.0/area(i))*(len2(i+1)*gy_1 - len2(i)*gy_0);
      }
      pcoord->Edge1Length(k,j  ,is,ie,len);
      pcoord->Edge1Length(k,j+1,is,ie,len_p1);
      for (int i=is; i<=ie; ++i) {
	dx1 = pcoord->dx1v(j);
        gx_0 = 0.25*(pgrav->phi(k,j,i+1)-pgrav->phi(k,j,i-1)+pgrav->phi(k,j-1,i+1)-pgrav->phi(k,j-1,i-1))/dx1;
        gx_1 = 0.25*(pgrav->phi(k,j,i+1)-pgrav->phi(k,j,i-1)+pgrav->phi(k,j+1,i+1)-pgrav->phi(k,j+1,i-1))/dx1;
        user_out_var(2,k,j,i) += (1.0/area(i))*(len_p1(i)*gx_1 - len(i)*gx_0);
      }
    }
  }

  //peos->PresFromRhoEs(den_curr_I, espec_curr_I);
  //pressure and espec outputs
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	Real den_curr = phydro->u(IDN,k,j,i);
        Real mom_sqr = SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i));
        Real mag_eng = 0.5*
                            (SQR(0.5*(pfield->b.x1f(k,j,i)+pfield->b.x1f(k  ,j  ,i+1)))
                            +SQR(0.5*(pfield->b.x2f(k,j,i)+pfield->b.x2f(k  ,j+1,i  )))
                            +SQR(0.5*(pfield->b.x3f(k,j,i)+pfield->b.x3f(k+1,j  ,i  ))));
	Real espec_curr = (phydro->u(IEN,k,j,i) - 0.5*(mom_sqr/den_curr) - mag_eng)/den_curr;
	
	user_out_var(3,k,j,i) = espec_curr;
        user_out_var(4,k,j,i) = peos->PresFromRhoEs(den_curr,espec_curr);
      }
    }
  }

  //std::cout << en_test << std::endl;
  //std::cout << "here" << std::endl;
}

Real vector_pot(int component,
                Real my_x1, Real my_x2, Real my_x3,
                Real x1c, Real x2c, Real x3c,
                Real I0, Real r0, Real rsurf, Real c, Real angle) {

  // rotation
  Real x_rot = my_x1-x1c;
  Real y_rot = (my_x2-x2c)*std::cos(angle) + (my_x3-x3c)*std::sin(angle);
  Real z_rot = (my_x3-x3c)*std::cos(angle) - (my_x2-x2c)*std::sin(angle);
  
  //Real r_dis = std::sqrt(std::pow((my_x1-x1c),2)+std::pow((my_x2-x2c),2)+std::pow((my_x3-x3c),2));
 
  Real ax_rot = -1.*((I0 * PI*std::pow(r0, 2.) * y_rot
                     * (1. + (15.*std::pow(r0, 2.) * (std::pow(r0, 2.)
                              + std::pow(x_rot, 2.) + std::pow(y_rot, 2.))) /
                         (8.*std::pow(std::pow(r0, 2.) + std::pow(x_rot, 2.)
                                     + std::pow(y_rot, 2.)
                                     + std::pow(z_rot, 2.), 2.)))) /
                       (c * std::pow(std::pow(r0, 2.)
                        + std::pow(x_rot, 2.) + std::pow(y_rot, 2.)
                        + std::pow(z_rot, 2.), 1.5)));

  Real ay_rot = (I0 * PI*std::pow(r0, 2.) * x_rot
                 * (1. + (15*std::pow(r0, 2.) * (std::pow(r0, 2.)
                        + std::pow(x_rot, 2.) + std::pow(y_rot, 2.)))/
                    (8.*std::pow(std::pow(r0, 2.) + std::pow(x_rot, 2.)
                                 + std::pow(y_rot, 2.) + std::pow(z_rot, 2.), 2.)))) /
                   (c * std::pow(std::pow(r0, 2.)
                    + std::pow(x_rot, 2.) + std::pow(y_rot, 2.)
                    + std::pow(z_rot, 2.), 1.5));
 
  Real az_rot = 0;

  // vector potential
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
  //const Real gamma = pmb->peos->GetGamma();
  //const Real fac = 2.0*PI*std::sqrt(gamma)/dx;
  for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
    for (int j=pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
      for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
        //Real dxi = pmb->pcoord->dx1f(i);
        //Real vol = dxi*dxi*dxi
        //Real nj = fac*std::sqrt(pmb->phydro->w(IPR,k,j,i))/pmb->phydro->w(IDN,k,j,i);
        //njmin = std::min(njmin, nj);
        Real m_amount = vol*pmb->phydro->u(IDN,k,j,i);
        mass = std::max(mass, m_amount);
      }
    }
  }
  if (mass > m_refine)
    return 1;
  //if (njmin < njeans)
    //return 1;
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
        //prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
        prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
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
        //prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
        prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
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
        //prim(IPR,k,jl-j,i) = prim(IPR,k,jl,i);
        prim(IEN,k,jl-j,i) = prim(IEN,k,jl,i);
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
        //prim(IPR,k,ju+j,i) = prim(IPR,k,ju,i);
        prim(IEN,k,ju+j,i) = prim(IEN,k,ju,i);
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
        //prim(IPR,kl-k,j,i) = prim(IPR,kl,j,i);
        prim(IEN,kl-k,j,i) = prim(IEN,kl,j,i);
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
        //prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
        prim(IEN,ku+k,j,i) = prim(IEN,ku,j,i);
      }
    }
  }
  return;
}
