//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file planetary_eos_table.cpp
//! \brief implements functions in class EquationOfState for an EOS lookup table
//======================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/interp_table.hpp"
#include "../eos.hpp"

namespace {
Real dens_pow = -1.0;
//----------------------------------------------------------------------------------------
//! \fn Real GetEosData(EosTable *ptable, int kOut, Real var, Real rho)
//! \brief Gets interpolated data from EOS table assuming 'var' has dimensions
//!        of energy per volume.
inline Real GetEosData(EosTable *ptable, int kOut, Real var, Real rho) {
  Real x1 = std::log10(rho * ptable->rhoUnit);
  Real x2 = std::log10(var * ptable->esUnit) + dens_pow * x1;
  return std::pow((Real)10, ptable->table.interpolate(kOut, x2, x1));
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated gas pressure from density and specific internal energy
Real EquationOfState::PresFromRhoEs(Real rho, Real espec) {
  return GetEosData(ptable, 0, espec, rho);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return interpolated adiabatic sound speed squared from density and specific
//! internal energy
Real EquationOfState::AsqFromRhoEs(Real rho, Real espec) {
  return GetEosData(ptable, 1, espec, rho);
}
