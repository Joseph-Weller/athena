//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file const.cpp
//  \brief implementation of radiation integrators: constant radiation
//======================================================================================


// Athena++ headers
#include "../radiation.hpp"
#include "../../parameter_input.hpp"

// Class header
#include "rad_integrators.hpp"

RadIntegrator::RadIntegrator(Radiation *prad, ParameterInput *pin)
{
  pmy_rad = prad;
  rad_G0_ = pin->GetReal("problem", "G0");
}

RadIntegrator::~RadIntegrator() {}

void RadIntegrator::UpdateRadiation() {}
