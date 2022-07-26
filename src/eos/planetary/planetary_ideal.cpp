//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal.cpp
//! \brief implements ideal EOS in general EOS framework, mostly for debuging
//======================================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <limits>   // std::numeric_limits<float>::epsilon()
#include <sstream>
#include <stdexcept> // std::invalid_argument

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"


//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real espec)
//! \brief Return gas pressure given density and specific internal energy
Real EquationOfState::PresFromRhoEs(Real rho, Real espec) {
  return (gamma_ - 1.)*rho*espec;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoEs(Real rho, Real es[ec])
//! \brief Return adiabatic sound speed squared given density and specific internal energy
Real EquationOfState::AsqFromRhoEs(Real rho, Real espec) {
  return gamma_ * (gamma_ - 1.) * espec;
}
