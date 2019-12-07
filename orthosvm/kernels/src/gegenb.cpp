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

double weights(double x, double y, double alfa)
{
    // TODO: Add the case when alfa == -0.5 to evaluate Tchebyshev
    
    if (-0.5 < alfa || alfa <= 0.5) return 1.0;
    if (alfa > 0.5)
    {
        double term_1 = (1.0 - (x * x)) * (1.0 - (y * y));
        double result = std::pow(term_1, alfa - 0.5) + 0.1;

        return result;
    }
}