#include "gegenbauer.hpp"

double pochhammer(double x, int n)
{
    if (n == 0) return 1.0;
    else if (n < 0 || x == 0) return 0.0;

    double result = 1.0;

    for (int i = 0; i < n; i++)
    {
        result *= (x + i);
    }

    return result;
}

double gegenbauerc(double x, int degree, double alfa)
{
    if (degree == 0) return 1.0;
    else if (degree == 1) return 2.0 * alfa * x;
    else
    {
        double first_value = 1.0;
        double second_value = 2.0 * alfa * x;
        double result = 0.0;

        for (int i = 2; i <= degree; i++)
        {
            result = 2.0 * x * (i + alfa - 1.0) * second_value - (
                (i + 2.0 * alfa - 2.0) * first_value
            );
            result /= i;
            first_value = second_value;
            second_value = result;
        }

        return result;
    }
}