// Algorithm implementation -*- C++ -*-

// Copyright (C) 2001-2023 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

// License texts are distributed in form/feature/licenses/.
// Adapted for bounded CUDA scanline sectors from GCC 13 libstdc++ introsort
// and heap routines. Preserve their equal-key permutation: feature suppression
// makes that observable. This compatibility policy is library-version specific.
#pragma once
#ifdef __CUDACC__
#define FORM_SORT_HD __host__ __device__
#else
#define FORM_SORT_HD
#endif
namespace form::detail {
FORM_SORT_HD inline void legacySwap(int& a,int& b) {int t=a;a=b;b=t;}
FORM_SORT_HD inline void legacyAdjust(int* ids,const double* keys,int hole,int n,int value) {
  const int top=hole;int child=hole;
  while(child<(n-1)/2) {
    child=2*(child+1);
    if(keys[ids[child]]<keys[ids[child-1]])--child;
    ids[hole]=ids[child];hole=child;
  }
  if(n%2==0 && child==(n-2)/2) {
    child=2*(child+1);ids[hole]=ids[child-1];hole=child-1;
  }
  int parent=(hole-1)/2;
  while(hole>top && keys[ids[parent]]<keys[value]) {
    ids[hole]=ids[parent];hole=parent;parent=(hole-1)/2;
  }
  ids[hole]=value;
}
FORM_SORT_HD inline void legacyHeap(int* ids,const double* keys,int n) {
  if(n<2)return;
  for(int parent=(n-2)/2;parent>=0;--parent)legacyAdjust(ids,keys,parent,n,ids[parent]);
  while(n>1){--n;int value=ids[n];ids[n]=ids[0];legacyAdjust(ids,keys,0,n,value);}
}
// At most 1024 elements. Explicit stack replaces right recursion; its maximum
// occupancy is bounded by the introsort depth limit (20).
FORM_SORT_HD inline void legacySelectionSort(int* ids,const double* keys,int n) {
  if(n<2)return;
  int depth=0;for(int k=n;k>1;k/=2)depth+=2;
  int starts[32],ends[32],depths[32],pending=0,lo=0,hi=n;
  for(;;) {
    while(hi-lo>16) {
      if(!depth){legacyHeap(ids+lo,keys,hi-lo);break;}
      --depth;
      const int a=lo+1,b=lo+(hi-lo)/2,c=hi-1;
      int median;
      if(keys[ids[a]]<keys[ids[b]]) {
        if(keys[ids[b]]<keys[ids[c]])median=b;
        else median=keys[ids[a]]<keys[ids[c]]?c:a;
      } else if(keys[ids[a]]<keys[ids[c]])median=a;
      else median=keys[ids[b]]<keys[ids[c]]?c:b;
      legacySwap(ids[lo],ids[median]);
      int left=lo+1,right=hi;
      for(;;) {
        while(keys[ids[left]]<keys[ids[lo]])++left;
        --right;while(keys[ids[lo]]<keys[ids[right]])--right;
        if(left>=right)break;
        legacySwap(ids[left],ids[right]);++left;
      }
      starts[pending]=lo;ends[pending]=left;depths[pending]=depth;++pending;
      lo=left; // Process right first, then the deferred left range.
    }
    if(!pending)break;
    --pending;lo=starts[pending];hi=ends[pending];depth=depths[pending];
  }
  const int guarded=n<16?n:16;
  for(int i=1;i<n;++i) {
    int value=ids[i],j=i;
    if(i<guarded && keys[value]<keys[ids[0]]) {
      while(j>0){ids[j]=ids[j-1];--j;}
    } else {
      while(keys[value]<keys[ids[j-1]]){ids[j]=ids[j-1];--j;}
    }
    ids[j]=value;
  }
}
}
#undef FORM_SORT_HD
