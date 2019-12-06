#include "gegenbauer.hpp"

double pochhammer(double x, int n)
{
    if (n == 0) return 1.0;
    else if (n < 0 || x == 0) return 0.0;

    double result = 0.0;

    for (int i = 0; i < n; i++)
    {
        result *= (x + i);
    }

    return result;
}