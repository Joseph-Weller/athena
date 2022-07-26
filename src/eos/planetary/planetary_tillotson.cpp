//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file planetary_tillotson.cpp
//! \brief implements Tillotson EOS in planetary EOS framework
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

// Tillotson params (cgs) for granite
namespace {
const Real lowa = 0.5;
const Real lowb = 1.3;
const Real alpha = 5.;
const Real beta = 5.;
const Real rho0 = 2.7;
const Real upA = 1.8e11;
const Real upB = 1.8e11;
const Real e0 = 1.6e11;
const Real eiv = 3.5e10;
const Real ecv = 1.8e11;
const Real pfloor = 1.e0;
const Real csqfloor = 1.e8;
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real espec)
//! \brief Return gas pressure given density and specific internal energy
//!  TODO(@pdmullen): this assumes CGS units.  Consider adding density/espec/pressure
//!  units so that we can generalize Tillotson to arbirtrary units

Real EquationOfState::PresFromRhoEs(Real rho, Real espec) {
  Real pres = pfloor;
  if (rho >= rho0 || espec < eiv) {
    pres = rho*espec*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/
           (e0*std::pow(rho,2.))))+(-1.+rho/rho0)*upA+std::pow(-1.+rho/rho0,2.)*upB;
  } else if (espec > ecv) {
    pres = rho*espec*lowa+((espec*lowb*rho)/(1.+(espec*std::pow(rho0,2.))/
           (e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*std::exp(-1.*(-1.+rho0/rho)*beta))*
           std::exp(-1.*std::pow(-1.+rho0/rho,2.)*alpha);
  } else {
    Real pc = std::max(
           rho*espec*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/
           (e0*std::pow(rho,2.))))+(-1.+rho/rho0)*upA+std::pow(-1.+rho/rho0,2.)*upB,
                       0.);
    Real pe = std::max(
           rho*espec*lowa+((espec*lowb*rho)/(1.+(espec*std::pow(rho0,2.))/
           (e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*std::exp(-1.*(-1.+rho0/rho)*beta))*
           std::exp(-1.*std::pow(-1.+rho0/rho,2.)*alpha),
                       0.);
    pres = ((espec-eiv)*pe + (ecv-espec)*pc)/(ecv-eiv);
  }
  return std::max(pres, pfloor);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoEs(Real rho, Real es[ec])
//! \brief Return adiabatic sound speed squared given density and specific internal energy
//!  TODO(@pdmullen): this assumes CGS units.  Consider adding density/espec/pressure
//!  units so that we can generalize Tillotson to arbirtrary units.  I have always *hated*
//!  this part of the code.  It might be the worst I have ever written.  Is there any way
//!  we can simplify this? The messiness here comes from the partial derivatives of
//!  P(rho, es) necessary in the sound speed calcuation:
//!
//!  cs^2 = [\partial P/\partial rho] |_es + P/rho^2 [\partial P \partial es] |_rho
//!
//!  I am relying on Mathematica to convert to C++ code for me with the
//!  CForm[] function.  Previous giant impact works employing the Tillotson EOS have
//!  extrapolated between cold and expanded states for the sound speed calculation.  But
//!  this can lead to thermodynamic inconsistencies in the sense that the above equation
//!  is not satisfied everywhere in the EOS (it is much cleaner in code though...).
//!  Also, there are many duplicated pow and exp calls...this is just bad practice and
//!  bad for performance... Furthermore, some of the exp(-HUGE_NUMBER) calls will yield
//!  zero, but should we be handling these cases directly?

Real EquationOfState::AsqFromRhoEs(Real rho, Real espec) {
  Real csq = csqfloor;
  if (rho >= rho0 || espec < eiv) {
    Real pres = rho*espec*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/
                (e0*std::pow(rho,2.))))+(-1.+rho/rho0)*upA+std::pow(-1.+rho/rho0,2.)*upB;
    if (pres > 0.) {  // code generated with Mathematica
      csq = (2.*e0*std::pow(espec,2.)*lowb*std::pow(rho,2.)*std::pow(rho0,2.))/
            std::pow(e0*std::pow(rho,2.)+espec*std::pow(rho0,2.),2.)+espec*(lowa+(e0*lowb*
            std::pow(rho,2.))/(e0*std::pow(rho,2.)+espec*std::pow(rho0,2.)))+upA/rho0+(2.*
            (rho-rho0)*upB)/std::pow(rho0,2.)+((std::pow(e0,2.)*(lowa+lowb)*
            std::pow(rho,5.)+2.*e0*espec*lowa*std::pow(rho,3.)*std::pow(rho0,2.)+
            std::pow(espec,2.)*lowa*rho*std::pow(rho0,4.))*(espec*rho*(lowa+(e0*lowb*
            std::pow(rho,2.))/(e0*std::pow(rho,2.)+espec*std::pow(rho0,2.)))+((rho-rho0)*
            (rho0*(upA-upB)+rho*upB))/std::pow(rho0,2.)))/std::pow(e0*std::pow(rho,3.)+
            espec*rho*std::pow(rho0,2.),2.);
    } else {
      csq = csqfloor;
    }
  } else if (espec > ecv) {  // code generated with Mathematica
    csq = espec*lowa +(2.*alpha*rho0*(-1.+rho0/rho)*((espec*lowb*rho)/(1.+(espec*
          std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
          std::exp(-1.*beta*(-1.+rho0/rho))))*(std::exp(-1.*alpha*
          std::pow(-1.+rho0/rho,2.))*std::pow(rho,-2.))+((2.*std::pow(espec,2.)*lowb*
          std::pow(rho0,2.))/(e0*std::pow(rho,2.)*std::pow(1.+(espec*std::pow(rho0,2.))/
          (e0*std::pow(rho,2.)),2.))+(espec*lowb)/(1.+(espec*std::pow(rho0,2.))/(e0*
          std::pow(rho,2.)))+upA*(std::exp(-1.*beta*(-1.+rho0/rho))*
          std::pow(rho0,-1.))+(beta*(-1.+rho/rho0)*rho0*upA)*(std::exp(-1.*beta*
          (-1.+rho0/rho))*std::pow(rho,-2.)))*std::exp(-1.*alpha*
          std::pow(-1.+rho0/rho,2.))+((lowa*rho+(-1.*((espec*lowb*std::pow(rho0,2.))/
          (e0*rho*std::pow(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)),2.)))+
          (lowb*rho)/(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.))))*
          std::exp(-1.*alpha*std::pow(-1.+rho0/rho,2.)))*(espec*lowa*rho+((espec*lowb*
          rho)/(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
          std::exp(-1.*beta*(-1.+rho0/rho)))*std::exp(-1.*alpha*
          std::pow(-1.+rho0/rho,2.))))/std::pow(rho,2.);
  } else {
    Real pres = rho*espec*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/
                (e0*std::pow(rho,2.))))+(-1.+rho/rho0)*upA+std::pow(-1.+rho/rho0,2.)*upB;
    if (pres > 0.) {  // code generated with Mathematica
      csq = ((espec*lowa*rho-espec*rho*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.))))+(ecv-espec)*(-((espec*lowb*std::pow(rho0,2.))/(e0*rho*
            std::pow(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)),2.)))+rho*(lowa+
            lowb/(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)))))+(espec-eiv)*(lowa*
            rho+(-((espec*lowb*std::pow(rho0,2.))/(e0*rho*std::pow(1.+(espec*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.)),2.)))+(lowb*rho)/(1.+(espec*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.))))*std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.)))-(-1.+rho/rho0)*upA+((espec*lowb*rho)/(1.+(espec*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
            std::exp(-1.*beta*(-1.+rho0/rho)))*std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.))-std::pow(-1.+rho/rho0,2.)*upB)*((espec-eiv)*(espec*
            lowa*rho+((espec*lowb*rho)/(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*std::exp(-1.*beta*(-1.+rho0/rho)))*
            std::exp(-1.*alpha*std::pow(-1.+rho0/rho,2.)))+(ecv-espec)*(espec*rho*(lowa+
            lowb/(1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.))))+(-1.+rho/rho0)*upA+
            std::pow(-1.+rho/rho0,2.)*upB)))/(std::pow(ecv-eiv,2.)*std::pow(rho,2.))+
            ((espec-eiv)*(espec*lowa+(2.*alpha*rho0*(-1.+rho0/rho)*((espec*lowb*rho)/(1.+
            (espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
            std::exp(-1.*beta*(-1.+rho0/rho))))*(std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.))*std::pow(rho,-2.))+((2.*std::pow(espec,2.)*lowb*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.)*std::pow(1.+(espec*std::pow(rho0,2.))/
            (e0*std::pow(rho,2.)),2.))+(espec*lowb)/(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.)))+upA*(std::exp(-1.*beta*(-1.+rho0/rho))*std::pow(rho0,-1.))+
            (beta*(-1.+rho/rho0)*rho0*upA)*(std::exp(-1.*beta*(-1.+rho0/rho))*
            std::pow(rho,-2.)))*std::exp(-1.*alpha*std::pow(-1.+rho0/rho,2.)))+
            (ecv-espec)*((2.*std::pow(espec,2.)*lowb*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.)*std::pow(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.)),2.))+espec*(lowa+lowb/(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.))))+upA/rho0+(2.*(-1.+rho/rho0)*upB)/rho0))/(ecv-eiv);
    } else {  // code generated with Mathematica
      csq = ((espec-eiv)*(espec*lowa+(2.*alpha*rho0*(-1.+rho0/rho)*((espec*lowb*rho)/(1.+
            (espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
            std::exp(-1.*beta*(-1.+rho0/rho))))*(std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.))*std::pow(rho,-2.))+((2.*std::pow(espec,2.)*lowb*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.)*std::pow(1.+(espec*std::pow(rho0,2.))/
            (e0*std::pow(rho,2.)),2.))+(espec*lowb)/(1.+(espec*std::pow(rho0,2.))/(e0*
            std::pow(rho,2.)))+upA*(std::exp(-1.*beta*(-1.+rho0/rho))*std::pow(rho0,-1.))+
            (beta*(-1.+rho/rho0)*rho0*upA)*(std::exp(-1.*beta*(-1.+rho0/rho))*
            std::pow(rho,-2.)))*std::exp(-1.*alpha*std::pow(-1.+rho0/rho,2.))))/(ecv-eiv)+
            ((espec-eiv)*(espec*lowa*rho+((espec*lowb*rho)/(1.+(espec*std::pow(rho0,2.))/
            (e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*std::exp(-1.*beta*
            (-1.+rho0/rho)))*std::exp(-1.*alpha*std::pow(-1.+rho0/rho,2.)))*(((espec-eiv)*
            (lowa*rho+(-((espec*lowb*std::pow(rho0,2.))/(e0*rho*std::pow(1.+(espec*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.)),2.)))+(lowb*rho)/(1.+(espec*
            std::pow(rho0,2.))/(e0*std::pow(rho,2.))))*std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.))))/(ecv-eiv)+(espec*lowa*rho+((espec*lowb*rho)/
            (1.+(espec*std::pow(rho0,2.))/(e0*std::pow(rho,2.)))+((-1.+rho/rho0)*upA)*
            std::exp(-1.*beta*(-1.+rho0/rho)))*std::exp(-1.*alpha*
            std::pow(-1.+rho0/rho,2.)))/(ecv-eiv)))/((ecv-eiv)*std::pow(rho,2.));
    }
  }
  return std::max(csq, csqfloor);
}
