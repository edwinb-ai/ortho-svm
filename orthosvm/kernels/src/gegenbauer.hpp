#ifndef GEGENBAUER_H
#define GEGENBAUER_H

#include <cstdio>

double pochhammer(double x, int n);
double gegenbauerc(double x, int degree, double alfa);
double kernel(double x, double y, int degree);

#endif // GEGENBAUER_H
