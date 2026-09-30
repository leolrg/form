// Adapted from Eigen 3.4's SelfAdjointEigenSolver.h and Tridiagonalization.h.
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Local CUDA adaptation: retain Eigen's optimized real 3x3 reduction, which
// lacks EIGEN_DEVICE_FUNC in Eigen 3.4, and reuse its device-annotated QR step.
#pragma once

#include <Eigen/Eigenvalues>
#include <limits>

namespace form::cuda_detail {
template<class T>
__device__ bool normalEigenvector(const Eigen::Matrix<T,3,3>& covariance,
                                 Eigen::Matrix<T,3,1>& normal) {
  Eigen::Matrix<T,3,3> mat=covariance.template triangularView<Eigen::Lower>();
  T scale=mat.cwiseAbs().maxCoeff();
  if(scale==T(0)) scale=T(1);
  mat.template triangularView<Eigen::Lower>()/=scale;
  Eigen::Matrix<T,3,1> diag;
  Eigen::Matrix<T,2,1> subdiag;
  diag[0]=mat(0,0);
  const T v1norm2=Eigen::numext::abs2(mat(2,0));
  if(v1norm2<=std::numeric_limits<T>::min()) {
    diag[1]=mat(1,1); diag[2]=mat(2,2);
    subdiag[0]=mat(1,0); subdiag[1]=mat(2,1);
    mat.setIdentity();
  } else {
    const T beta=sqrt(Eigen::numext::abs2(mat(1,0))+v1norm2);
    const T invBeta=T(1)/beta;
    const T m01=mat(1,0)*invBeta, m02=mat(2,0)*invBeta;
    const T q=T(2)*m01*mat(2,1)+m02*(mat(2,2)-mat(1,1));
    diag[1]=mat(1,1)+m02*q; diag[2]=mat(2,2)-m02*q;
    subdiag[0]=beta; subdiag[1]=mat(2,1)-m01*q;
    mat << 1,0,0, 0,m01,m02, 0,m02,-m01;
  }
  const auto info=Eigen::internal::computeFromTridiagonal_impl(
      diag,subdiag,Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>>::m_maxIterations,true,mat);
  normal=mat.col(0);
  normal.normalize();
  return info==Eigen::Success && isfinite(normal.x()) && isfinite(normal.y()) && isfinite(normal.z());
}
}
