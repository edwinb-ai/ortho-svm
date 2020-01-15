# Orthogonal Polynomial Kernels for Support Vector Machines [![Build Status](https://travis-ci.org/edwinb-ai/orthosvm.svg?branch=master)](https://travis-ci.org/edwinb-ai/orthosvm) [![Build status](https://ci.appveyor.com/api/projects/status/1an2ff4ouug8pp8x/branch/master?svg=true)](https://ci.appveyor.com/project/edwinb-ai/orthosvm/branch/master)

These are the implementations for the **orthogonal polynomial kernels**
to be used in *Support Vector Machines* (SVMs) as described in the paper [Padierna et al, 2018](https://www.sciencedirect.com/science/article/abs/pii/S0031320318302280).

The main idea is to employ several orthogonal polynomial descriptions such as Hermite and Gegenbauer families of polynomials
and build fully fledged Mercer kernels to use in classification and regression tasks with SVMs.