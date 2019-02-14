#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.hpp
//  \brief defines Gravity class which implements data and functions for gravitational
//         potential. Shared by both Multigrid and FFT schemes for self-gravity.

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/cc/bvals_cc.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;
class GravityBoundaryValues;

//! \class Gravity
//  \brief gravitational potential data and functions

class Gravity {
 public:
  Gravity(MeshBlock *pmb, ParameterInput *pin);
  ~Gravity();

  MeshBlock* pmy_block;  // ptr to MeshBlock containing this Field

  //GravityBoundaryVariable *pgbval;
  // KGF: temporary workaround:
  CellCenteredBoundaryVariable *pgbval;

  AthenaArray<Real> phi;  // gravitational potential
  Real gconst, four_pi_G;
  Real grav_mean_rho;
  bool srcterm;

  void Initialize(ParameterInput *pin);
  void Solver(const AthenaArray<Real> &u);

 private:
  bool gravity_tensor_momentum_;
  bool gravity_tensor_energy_;
};

#endif // GRAVITY_GRAVITY_HPP_
