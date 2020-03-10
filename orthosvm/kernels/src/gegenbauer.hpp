#ifndef GEGENBAUER_H
#define GEGENBAUER_H

#include <cmath>

double pochhammer(double x, int n);
double gegenbauerc(double x, int degree, double alfa);
double weights(double x, double y, double alfa, int k);
double kernel(double x, double y, int degree, double alfa);

#endif // GEGENBAUER_H
