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

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
    {
      AllocateUserOutputVariables(4);
      return;
    }

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  
  Real R_max = pin->GetOrAddReal("problem", "r_max", 1.0);
  //std::cout << "R_max " << R_max << std::endl;
  //Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real ea   = pin->GetOrAddReal("problem", "eamb", 1.0); 
  Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  Real dc = pin->GetOrAddReal("problem", "dcent", 1.0);
  Real xvel = pin->GetOrAddReal("problem", "vx", 0.0);
  Real yvel = pin->GetOrAddReal("problem", "vy", 0.0);
  Real pulse_var = pin->GetOrAddReal("problem", "pulse", 1.0);
  Real Radius_Earth = 6.371e8;  
  Real eoff   = pin->GetOrAddReal("problem", "e_offset", 0.0);
  Real merge = pin->GetOrAddReal("problem", "e_merge", 0.0);

  Real gamma = pin->GetReal("hydro","gamma");
  Real gm1 = gamma - 1.0;
  
  //Real ecent = pin->GetOrAddReal("problem", "ecent", 1.0); //update to read input
  Real ecent = pin->GetOrAddReal("problem", "ecent", 1.0);
  //Real pcent = (2*gconst/PI)*dc*dc*R_max*R_max;

  Real den_stab[10000];
  Real espec_stab[10000];
  Real r_store[10000];
  Real r0 = 0.01;
  Real krho1 = 0.0;
  Real krho2 = 0.0;
  Real kespec1 = 0.0;
  Real kespec2 = 0.0;
  Real k1M = 0.0;
  Real k2M = 0.0;
  Real cs_2 = 0.0;
  //Real pres = peos->PresFromRhoEs(den_curr, espec_curr);
  
  Real r0_2 = 0.0;
  Real den_curr = dc;
  Real espec_curr = ecent;
  Real dr = R_max/10000;
  Real drho = 0.0;
  Real despec = 0.0;
  Real dM = 0.0;
  int place = 0;
  Real M = 0.0;
  Real pres = peos->PresFromRhoEs(den_curr, espec_curr);

  Real espec_max = 0.0;
  Real en_max = 0.0;

  //Real xvel = 0.0;
  //Real yvel = 0.0;
  //while (pres >0){
  while(r0<R_max){
    den_stab[place] = den_curr;
    espec_stab[place] = espec_curr;
    r_store[place]= r0;

    cs_2 = peos->AsqFromRhoEs(den_curr, espec_curr);
    pres = peos->PresFromRhoEs(den_curr, espec_curr);
    drho = -1*gconst*M*den_curr/(cs_2*std::pow(r0, 2));
    despec = (pres/pow(den_curr, 2))*drho;
    dM = 4*PI*den_curr*pow(r0,2);

    den_curr = den_curr + dr*drho;
    espec_curr = espec_curr + dr*despec;
    M = M + dr*dM;
    r0 = r0+dr;
    place = place +1;

    //krho1 = den_curr +0.5*dr*drho;
    //kespec1 = espec_curr +0.5*dr*despec;
    //k1M = M +0.5*dr*dM;
    //r0_2 = r0 + 0.5*dr;

    //cs_2 = peos->AsqFromRhoEs(krho1, kespec1);
    //pres = peos->PresFromRhoEs(krho1, kespec1);
    //drho = -1*gconst*k1M*krho1/(cs_2*std::pow(r0_2, 2));
    //despec = (pres/pow(krho1, 2))*drho;
    //dM = 4*PI*krho1*pow(r0_2,2);

    //den_curr = den_curr + dr*drho;
    //espec_curr = espec_curr + dr*despec;
    //M = M + dr*dM;
    //r0 = r0+dr;
    //place = place +1;
  }
  //std::cout << espec_stab[0] << std::endl;
  //std::cout << espec_stab[5000] << std::endl;

     //implement rk2 step for tillotson eos
     //k1 = del_r*func(var_array, r, G, gam)
     //k2 = del_r*func(var_array+0.5*k1, r+0.5*del_r, G, gam)
     //cs_2 = til_cs(rho_val, e_val)
     //Pres = til_pres(rho_val, e_val)
     //drho = -G*M_val*rho_val/(cs_2*r**2)
     //despec = fe = (Pres/rho_val**2)*frho
     //dM = 4*np.pi*rho_val*r**2
     //den_stab = den_stab + krho2
     //espec_stab = espec_stab +kespec2
     //M = M + k2M

  Real atm_merge = pin->GetOrAddReal("problem", "atm_merge", 0.02);
  Real atm_ext = pin->GetOrAddReal("problem", "atm_ext", 1.25);
  Real Poly_cut = 1.0-atm_merge;
  
  Real orb_vel = std::sqrt(gconst*M/R_max);
  Real spin = pin->GetOrAddReal("problem", "spin", 0.5);

  // get coordinates of center of bump, and convert to Cartesian if necessary
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
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        }

        Real den = 0.0;
	Real den_I = 0.0;
	Real den_back = da;
        Real den_spot = 0.0;
        Real espec = 0.0;
	Real espec_I = 0.0;
	Real espec_back = ea;
        Real espec_spot = 0.0;
        Real momx = 0.0;
        Real momy = 0.0;
        Real kin  = 0.0;
	int spot = 0;
	Real r_spot = 0.0;
	Real x = pcoord->x1v(i);
	Real y = pcoord->x2v(j);
	//xvel = 0.0;
	//yvel = 0.0;
	Real rad_2d = std::sqrt(SQR(x - x0) + SQR(y - y0));
	//std::cout << position << std::endl;
        if (rad < Radius_Earth*atm_ext) {
          if (rad < R_max*Poly_cut) {
	//if (rad < R_max) {
	    while (r_spot <rad){
	      r_spot = r_spot + dr;
	      spot = spot + 1;
	    }
	    //std::cout << "here if before error, otherwise it won't print" << std::endl;
	    //std::cout << "current rad" << rad <<"current rspot"<< r_spot<< std::endl;
	    if (rad < 0.01){
	      den_spot = dc;
              espec_spot = ecent;
	    } else{
	      den_spot = den_stab[(spot-1)]+(rad - r_store[(spot-1)])*(den_stab[spot]-den_stab[(spot-1)])/(r_store[spot]-r_store[(spot-1)]);
	      espec_spot = espec_stab[(spot-1)]+(rad - r_store[(spot-1)])*(espec_stab[spot]-espec_stab[(spot-1)])/(r_store[spot]-r_store[(spot-1)]);
	      //from stored value interpolate to point
	    }
	  //den_stab = 1;
	  //den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);
          //std::exp(1/(std::pow(rcrit, 2))-1/(std::pow((rad - rcrit), 2)));
            den_I = den_spot;
	    if (merge < 0.5){
              espec_I = espec_spot+eoff;
	    } else{
              espec_I = espec_spot+eoff*(2.0)*(rad*rad)/(R_max*R_max + rad*rad);
	    }
	  //std::cout << espec << std::endl;
	  //bump function adds density within critical radius
            espec_max = std::max(espec_max, espec);
	    en_max = std::max(en_max, espec*den);
	  //pres_pol = pcent*std::pow(R_max/(PI*rad)*std::sin((PI*rad)/R_max),2);
          //pres = pres_pol;
	  
	  //add spin
	    Real xvel_new = xvel - spin*orb_vel*(y-y0)/R_max;
            Real yvel_new = yvel + spin*orb_vel*(x-x0)/R_max;
	  
	    //xvel = xvel_input - spin*orb_vel*(y-y0)/(R_max);
	    //yvel = yvel_input + spin*orb_vel*(x-x0)/(R_max);

            momx = den_I*xvel_new;
            momy = den_I*yvel_new;
            //kin = 0.5*den*xvel_new*xvel_new+0.5*den*yvel_new*yvel_new;
          } else {
            //while (r_spot < R_max*Poly_cut){
            //  r_spot = r_spot + dr;
            //  spot = spot + 1;
            //}
            //std::cout << r_spot << std::endl;
            //den = den_stab[(spot-1)] * std::pow(R_max*Poly_cut/rad,15.0);

            //espec = espec_stab[(spot-1)] * R_max*Poly_cut/rad;
            
	    //Real targ_amb = 0.1;
            //den_E = targ_amb*exp(-1*(rad-R_max)/(0.057*Radius_Earth));

	    //espec_E = 1.0e11*exp(-1*(rad-R_max)/(1.907*Radius_Earth));
            Real imp_amb = 0.1;
            den_I = imp_amb*exp(-1*(rad-R_max)/(0.0509*Radius_Earth));


            espec_I = 7.5e10*exp(-1*(rad-R_max)/(1.568*Radius_Earth))
                    + 4.0e9*exp((rad-1.2*Radius_Earth)/(0.0932*Radius_Earth));


	    //add spin
	    
	    Real xvel_new = xvel - spin*orb_vel*(y-y0)/R_max;
            Real yvel_new = yvel + spin*orb_vel*(x-x0)/R_max;
            
	    //xvel = xvel_input - spin*orb_vel*(y-y0)/(R_max);
            //yvel = yvel_input + spin*orb_vel*(x-x0)/(R_max);

            momx = den_I*xvel_new;
            momy = den_I*yvel_new;
            //kin = 0.5*den*xvel_new*xvel_new+0.5*den*yvel_new*yvel_new;
          }

        }
	den = den_back + den_I;
	espec = (espec_back*den_back + espec_I*den_I)/den;
	//std::cout << position << std::endl;
        position = position +1;
	//std::cout << kin << std::endl;

	//testing if something is wrong with assignment
	//phydro->u(IDN,k,j,i) = da;
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = momx;
        phydro->u(IM2,k,j,i) = momy;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = den*espec*pulse_var + 0.5*momx*momx/den+0.5*momy*momy/den;
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

    Real x1c_target   = 0.0;
    Real x2c_target   = 0.0;
    Real x3c_target   = 0.0;

    Real x1c_impactor = 0.0;
    Real x2c_impactor = 0.0;
    Real x3c_impactor = 0.0;

    // Dipole parameters
    Real c              = pin->GetReal("problem","c");
    Real I0_target      = pin->GetReal("problem","I0_target");
    Real I0_impactor    = 0.0;
    Real rsurf_target   = R_max;
    Real rsurf_impactor = 0.0;
    Real angle_target   = pin->GetReal("problem","angle_target")*PI/180.;
    Real angle_impactor = 0.0;
    Real r0_target      = R_max/3.0;
    Real r0_impactor    = 0.0;

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

        a1_impactor   = 0.0;
      } else {
        a1_target   = vector_pot(X1DIR,
                                 pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a1_impactor = 0.0;
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

        a2_impactor   = 0.0;
      } else {
        a2_target   = vector_pot(X2DIR,
                                 pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a2_impactor = 0.0;
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

        a3_impactor   = 0.0;

      } else {
        a3_target   = vector_pot(X3DIR,
                                 pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                 x1c_target, x2c_target, x3c_target,
                                 I0_target, r0_target,
                                 rsurf_target, c, angle_target);

        a3_impactor = 0.0;
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
 //std::cout << "here" << std::endl;
 //gets here
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pfield->b;
   
  Real en_test = 0.0;
  Real espec_test = 0.0; 
  //std::cout << "here" << std::endl;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pcoord->Face3Area(k,   j,   is, ie,   face3m);    
      for(int i=is; i<=ie; i++) {
        //std::cout << b.x2f(k,j,i+1) << std::endl;
	//user_out_var(0,k,j,i) = (face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
        //      +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
        //      +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
	Real den_curr = phydro->u(IDN,k,j,i);
	Real mom_sqr = SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i));
	Real espec_curr = (phydro->u(IEN,k,j,i) - 0.5*(mom_sqr/den_curr))/den_curr;
	//std::cout << mom_sqr << std::endl;
	user_out_var(0,k,j,i) = peos->PresFromRhoEs(den_curr, espec_curr);
	user_out_var(1,k,j,i) = espec_curr;
	user_out_var(3,k,j,i) = peos->AsqFromRhoEs(den_curr, espec_curr);
	//std::cout << "here" << std::endl;
	espec_test = std::max(espec_test, espec_curr);
	en_test = std::max(phydro->u(IEN,k,j,i), en_test);
        //std::cout << phydro->u(IEN,k,j,i) << std::endl;
      }
    }
  }
  AthenaArray<Real> area, len2, len, len_p1;
  //z vorticity calc
  Real vi_low = 0.0;
  Real vi_high = 0.0;
  Real vj_low = 0.0;
  Real vj_high = 0.0;

  area.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len2.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len.NewAthenaArray ((ie-is)+2*NGHOST+2);
  len_p1.NewAthenaArray ((ie-is)+2*NGHOST+2);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face3Area(k,j,is,ie,area);
      pcoord->Edge2Length(k,j,is,ie+1,len2);

      pcoord->Edge1Length(k,j  ,is,ie,len);
      pcoord->Edge1Length(k,j+1,is,ie,len_p1);
      for (int i=is; i<=ie; ++i) {
        vi_high = 0.5*((phydro->u(IM2,k,j,i+1)/phydro->w(IDN,k,j,i+1))+(phydro->u(IM2,k,j,i)/phydro->w(IDN,k,j,i)));
        vi_low  = 0.5*((phydro->u(IM2,k,j,i)/phydro->w(IDN,k,j,i))+(phydro->u(IM2,k,j,i-1)/phydro->w(IDN,k,j,i-1)));

        vj_high = 0.5*((phydro->u(IM1,k,j+1,i)/phydro->w(IDN,k,j+1,i))+(phydro->u(IM1,k,j,i)/phydro->w(IDN,k,j,i)));
        vj_low  = 0.5*((phydro->u(IM1,k,j,i)/phydro->w(IDN,k,j,i))+(phydro->u(IM1,k,j-1,i)/phydro->w(IDN,k,j-1,i)));

        user_out_var(2,k,j,i) = ((1.0/area(i))*(len2(i+1)*vi_high - len2(i)*vi_low)) - (1.0/area(i))*(len_p1(i)*vj_high - len(i)*vj_low);
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
