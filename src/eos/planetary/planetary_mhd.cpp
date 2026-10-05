//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file planetary_mhd.cpp
//! \brief implements most but not all of the functions in class
//! EquationOfState for planetary EOS MHD
//!
//! These functions MUST be implemented in an additional file.
//!
//! Real EquationOfState::PresFromRhoEs(Real rho, Real espec)
//! Real EquationOfState::AsqFromRhoEs(Real rho, Real espec)


// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../field/field.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) :
  ptable{pmb->pmy_mesh->peos_table},
  pmy_block_{pmb},
  gamma_{pin->GetOrAddReal("hydro", "gamma", 2.)},
  max_velocity_{pin->GetOrAddReal("hydro", "max_velocity", std::numeric_limits<Real>::max())},
  specific_intenergy_ceiling_{pin->GetOrAddReal("hydro", "eceiling", std::numeric_limits<Real>::max())},
  density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min))},
  espec_floor_{pin->GetOrAddReal("hydro", "efloor", std::sqrt(1024*float_min))},
  scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))} {
  if (EOS_TABLE_ENABLED) {
    if (!ptable) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::EquationOfState" << std::endl
          << "EOS table data uninitialized. Should be initialized by Mesh." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il,int iu, int jl,int ju, int kl,int ku) {

  pmy_block_->pfield->CalculateCellCenteredField(b,bcc,pco,il,iu,jl,ju,kl,ku);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        Real& w_d  = prim(IDN,k,j,i);
        Real& w_vx = prim(IVX,k,j,i);
        Real& w_vy = prim(IVY,k,j,i);
        Real& w_vz = prim(IVZ,k,j,i);
        Real& w_e  = prim(IEN,k,j,i);

	// modify floors based on distance to orgin
	//
	Real x = pco->x1v(i);
        Real y = pco->x2v(j);
        Real z = pco->x3v(k);

	Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
	Real den_floor_new = density_floor_;
	Real espec_floor_new = espec_floor_;
	Real R_E = 6.371e8;

	//if(r < 5.0*R_E){
        //  den_floor_new = 10.0*density_floor_;
	//  espec_floor_new = espec_floor_;
	//} else {
	//  den_floor_new = std::max(density_floor_, 10.0*density_floor_ - (9*density_floor_/(35.0*R_E))*(r-5.0*R_E));
	//  espec_floor_new = std::min(10.0*espec_floor_, espec_floor_ + (9*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
	//}

	if(r < 5.0*R_E){
          den_floor_new = 100.0*density_floor_;
          espec_floor_new = espec_floor_;
        } else {
          den_floor_new = std::max(density_floor_, 100.0*density_floor_/(1.0+99.0*SQR((r-5.0*R_E)/(35.0*R_E))));
          espec_floor_new = std::min(5.0*espec_floor_, espec_floor_ + (4.0*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
        }


        // apply density floor, without changing momentum or energy
        //u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        u_d = (u_d > den_floor_new) ?  u_d : den_floor_new;
	w_d = u_d;

        Real di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        const Real& bcc1 = bcc(IB1,k,j,i);
        const Real& bcc2 = bcc(IB2,k,j,i);
        const Real& bcc3 = bcc(IB3,k,j,i);

        Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
        Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
	//attempted fix below
	w_e = di*(u_e - ke - pb);

        // apply specific internal energy floor, correct total energy
        //u_e = (w_e > espec_floor_) ? w_e : (w_d*espec_floor_ + ke + pb);
        //attempted fix below
	//u_e = (w_e > espec_floor_) ? u_e : (w_d*espec_floor_ + ke + pb);
	u_e = (w_e > espec_floor_new) ? u_e : (w_d*espec_floor_new + ke + pb);
	//u_e = (w_e > espec_floor_) ? w_e : espec_floor_;
	//attempated fix below
        //w_e = (w_e > espec_floor_) ? w_e : espec_floor_;
	w_e = (w_e > espec_floor_new) ? w_e : espec_floor_new;

	//apply velocity ceiling
	Real m_sq = SQR(u_m1) + SQR(u_m2) + SQR(u_m3);
        Real v_sq = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
        Real m_abs = std::sqrt(m_sq);
        Real v_abs = (std::sqrt(v_sq) > 0.0) ? std::sqrt(v_sq) : 0.0;

	if (v_abs > max_velocity_) {
          Real v_over_m = max_velocity_ / m_abs;

          // apply velocity ceiling
          Real tmp_vx = u_m1 * v_over_m;
          Real tmp_vy = u_m2 * v_over_m;
          Real tmp_vz = u_m3 * v_over_m;

          // correct momentum
          u_m1 = w_d * tmp_vx;
          u_m2 = w_d * tmp_vy;
          u_m3 = w_d * tmp_vz;

          // correct velocities
          w_vx = u_m1*di;
          w_vy = u_m2*di;
          w_vz = u_m3*di;

          // correct kinetic energy
          Real delta_ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) - ke;
          ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));

          // correct total energy
          u_e += delta_ke;
	  //is above neccessary? shouldn't change w_e
	  //u_e = w_d*w_e + ke + pb;

          // recalculate specific internal energy
	  //is below neccessary? shouldn't change we
	  w_e = di*(u_e - ke - pb);

          // reapply specific internal energy floor
          //u_e = (w_e > espec_floor_) ? u_e : (w_d*espec_floor_ + ke + pb);
	  u_e = (w_e > espec_floor_new) ? u_e : (w_d*espec_floor_new + ke + pb);
          //u_e = (w_e > espec_floor_) ? w_e : espec_floor_;
          //attempated fix below
          //w_e = (w_e > espec_floor_) ? w_e : espec_floor_;
	  w_e = (w_e > espec_floor_new) ? w_e : espec_floor_new;
        }

	// apply specific internal energy ceiling
        if (w_e > specific_intenergy_ceiling_) {
          // correct total energy
          u_e = w_d*specific_intenergy_ceiling_ + ke + pb;

          // recalculate specific internal energy
          w_e = di * (u_e - ke - pb);
        }


      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  // Force outer-loop vectorization
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      //#pragma omp simd
#pragma novector
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);
        //const Real& w_p  = prim(IPR,k,j,i);
        // added below line to try to correct error
	const Real& w_e  = prim(IEN,k,j,i);

        const Real& bcc1 = bc(IB1,k,j,i);
        const Real& bcc2 = bc(IB2,k,j,i);
        const Real& bcc3 = bc(IB3,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
        // cellwise conversion
        u_e = w_d*w_e + 0.5*(w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz))
                             + (SQR(bcc1) + SQR(bcc2) + SQR(bcc3)));

	//adding max velocity correction here as well

        //Real v_sq = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
        //Real v_abs = (std::sqrt(v_sq) > 0.0) ? std::sqrt(v_sq) : 0.0;
        //if (v_abs > max_velocity_) {
        //  Real vm_over_v = max_velocity_ / v_abs;

          // apply velocity ceiling
          //Real tmp_vx = w_vx * vm_over_v;
          //Real tmp_vy = w_vy * vm_over_v;
          //Real tmp_vz = w_vz * vm_over_v;

          // correct momentum/energy
	  //u_m1 = tmp_vx*w_d;
          //u_m2 = tmp_vy*w_d;
          //u_m3 = tmp_vz*w_d;
          // cellwise conversion
          //u_e = w_d*w_e + 0.5*(w_d*(SQR(tmp_vx) + SQR(tmp_vy) + SQR(tmp_vz))
          //                   + (SQR(bcc1) + SQR(bcc2) + SQR(bcc3)));

        //}
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
//! \brief returns adiabatic sound speed given vector of primitive variables

Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(AsqFromRhoEs(prim[IDN], prim[IEN]));
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::FastMagnetosonicSpeed(const Real prim[], const Real bx)
//! \brief returns fast magnetosonic speed given vector of primitive variables
//! Note the formula for (C_f)^2 is positive definite, so this func never returns a NaN
Real EquationOfState::FastMagnetosonicSpeed(const Real prim[(NWAVE)], const Real bx) {
  Real asq = AsqFromRhoEs(prim[IDN], prim[IEN]) * prim[IDN]; // Actually rho*asq
  Real vaxsq = bx*bx;
  Real ct2 = (prim[IBY]*prim[IBY] + prim[IBZ]*prim[IBZ]);
  Real qsq = vaxsq + ct2 + asq;
  Real tmp = vaxsq + ct2 - asq;
  return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/prim[IDN]);
}

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 int i)
//! \brief Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_e  = prim(IEN,i);

  //adding location dependent floors

  Real x = pmy_block_->pcoord->x1v(i);
  Real y = pmy_block_->pcoord->x2v(j);
  Real z = pmy_block_->pcoord->x3v(k);

  Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real den_floor_new = density_floor_;
  Real espec_floor_new = espec_floor_;
  Real R_E = 6.371e8;

  //if(r < 5.0*R_E){
  //  den_floor_new = 10.0*density_floor_;
  //  espec_floor_new = espec_floor_;
  //} else {
  //  den_floor_new = std::max(density_floor_, 10.0*density_floor_ - (9*density_floor_/(35.0*R_E))*(r-5.0*R_E));
  //  espec_floor_new = std::min(10.0*espec_floor_, espec_floor_ + (9*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
  //}

  if(r < 5.0*R_E){
    den_floor_new = 100.0*density_floor_;
    espec_floor_new = espec_floor_;
  } else {
    den_floor_new = std::max(density_floor_, 100.0*density_floor_/(1.0+99.0*SQR((r-5.0*R_E)/(35.0*R_E))));
    espec_floor_new = std::min(5.0*espec_floor_, espec_floor_ + (4.0*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
  }


  // added below line to try to correct error
  //Real& w_p  = prim(IPR,i);

  // apply density floor
  //w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  w_d = (w_d > den_floor_new) ?  w_d : den_floor_new;
  // apply pressure floor
  //w_p = (w_e > espec_floor_) ?  w_e : espec_floor_;
  //attempted fix below
  //w_e = (w_e > espec_floor_) ?  w_e : espec_floor_;
  w_e = (w_e > espec_floor_new) ?  w_e : espec_floor_new;

  //
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply specific internal energy (prim) floor and correct energy (cons)
//! (typically after W(U))
void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_e  = prim(IEN,k,j,i);
  // added below line to try to correct error
  //Real& w_p  = prim(IPR,k,j,i);

  Real& u_d  = cons(IDN,k,j,i);
  Real& u_e  = cons(IEN,k,j,i);
  const Real& bcc1 = bcc(IB1,k,j,i);
  const Real& bcc2 = bcc(IB2,k,j,i);
  const Real& bcc3 = bcc(IB3,k,j,i);
  
  //adjusting density floor based on magnetic energy
  //Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
  //if (pb > 1.0e4) {
  //  w_d = (w_d > 10.0*density_floor_) ?  w_d : 10.0*density_floor_;
  //} else if (pb > 5.0e3) {
  //  w_d = (w_d > ((9.0/5.0e3)*(pb-5.0e3)+1.0)*density_floor_) ?  w_d : ((9.0/5.0e3)*(pb-5.0e3)+1.0)*density_floor_;
  //} else{
  //  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  //}


  //adding location dependent floors

  Real x = pmy_block_->pcoord->x1v(i);
  Real y = pmy_block_->pcoord->x2v(j);
  Real z = pmy_block_->pcoord->x3v(k);

  Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real den_floor_new = density_floor_;
  Real espec_floor_new = espec_floor_;
  Real R_E = 6.371e8;

  //if(r < 5.0*R_E){
  //  den_floor_new = 10.0*density_floor_;
  //  espec_floor_new = espec_floor_;
  //} else {
  //  den_floor_new = std::max(density_floor_, 10.0*density_floor_ - (9*density_floor_/(35.0*R_E))*(r-5.0*R_E));
  //  espec_floor_new = std::min(10.0*espec_floor_, espec_floor_ + (9*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
  //}

  if(r < 5.0*R_E){
    den_floor_new = 100.0*density_floor_;
    espec_floor_new = espec_floor_;
  } else {
    den_floor_new = std::max(density_floor_, 100.0*density_floor_/(1.0+99.0*SQR((r-5.0*R_E)/(35.0*R_E))));
    espec_floor_new = std::min(5.0*espec_floor_, espec_floor_ + (4.0*espec_floor_/(35.0*R_E))*(r-5.0*R_E));
  }
  
  // apply (prim) density floor, without changing momentum or energy

  //w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  w_d = (w_d > den_floor_new) ?  w_d : den_floor_new;
  // ensure cons density matches
  u_d = w_d;

  Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
  Real e_k = 0.5*w_d*(SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)));
  // apply pressure floor, correct total energy
  //u_e = (w_e > espec_floor_) ? u_e : w_d*espec_floor_ + e_k + pb;
  u_e = (w_e > espec_floor_new) ? u_e : w_d*espec_floor_new + e_k + pb;
  //w_p = (w_e > espec_floor_) ? w_e : espec_floor_;
  //attempted fix below
  //w_e = (w_e > espec_floor_) ? w_e : espec_floor_;
  w_e = (w_e > espec_floor_new) ? w_e : espec_floor_new;

  return;
}

Real EquationOfState::GetGamma() {
  std::stringstream msg;
  msg << "GetGamma is not defined for planetary EOS." << std::endl;
  ATHENA_ERROR(msg);
}
