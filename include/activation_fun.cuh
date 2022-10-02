///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cuh
// Description:  Header file for activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      September 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void noActMeanVar(float const *mz, float const *Sz, float *ma,
                             float *J, float *Sa, int zpos, int n);

__global__ void tanhMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n);

__global__ void sigmoidMeanVar(float const *mz, float const *Sz, float *ma,
                               float *J, float *Sa, int zpos, int n);

__global__ void reluMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n);

__global__ void softplusMeanVar(float const *mz, float const *Sz, float *ma,
                                float *J, float *Sa, int zpos, int n);

__global__ void leakyreluMeanVar(float const *mz, float const *Sz, float alpha,
                                 float *ma, float *J, float *Sa, int zpos,
                                 int n);

__global__ void exp_fun(float const *mz, float const *Sz, int n, float *ma,
                        float *Sa, float *Cza);

__global__ void actFullCov(float const *Szf, float const *J, int no, int B,
                           int zposOut, float *Saf);

__global__ void noActFullCov(float const *Szf, float *Saf, int Nf);