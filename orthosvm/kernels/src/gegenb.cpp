#include "gegenbauer.hpp"

double pochhammer(double x, int n)
{
    //  Compute the Pochhammer symbol (x)_n for rising factorials
    //  using the Gamma function like so:
    //  (x)_n = x(x+1)(x+2)...(x+n-1)

    // Special values for the Pochhammer symbol
    if (n == 0)
        return 1.0;
    else if (n < 0 || x == 0)
        return 0.0;

    double result = 1.0;

    for (int i = 0; i < n; i++)
    {
        result *= (x + i);
    }

    return result;
}

double gegenbauerc(double x, int degree, double alfa)
{
    //  Compute the Gegenbauer polynomials of degree `degree` and special parameter
    //  alfa using the 3-term recurrence relation

    //  Compute the special values, base cases
    if (degree == 0)
        return 1.0;
    else if (degree == 1)
        return 2.0 * alfa * x;
    else
    {
        double first_value = 1.0;
        double second_value = 2.0 * alfa * x;
        double result = 0.0;

        for (int i = 2; i <= degree; i++)
        {
            result = 2.0 * x * (i + alfa) * second_value;
            result -= (i + (2.0 * alfa) - 1.0) * first_value;
            result /= (i + 1.0);
            first_value = second_value;
            second_value = result;
        }

        return result;
    }
}

double weights(double x, double y, double alfa, int k)
{
    //  This computes the weight function (measure) for the Gegenbauer polynomial
    //  with special paramter alfa

    double result = 0.0;
    double weight_factor = 0.0;

    // Compute the Pochhammer symbol
    double term_1 = k + 1.0;
    double term_2 = pochhammer(2.0 * alfa + 1.0, k) / pochhammer(1.0, k);

    // Avoid rounding errors
    if (term_2 <= 1E-10)
    {
        term_2 = 0.0;
    }
    else
    {
        term_2 *= term_2;
    }

    //  A value between -0.5 and 0.5 is unity
    if (alfa <= 0.0)
    {
        result = 1.0;
    }
    else
    {
        weight_factor = (1.0 - (x * x)) * (1.0 - (y * y));
        // We need to add an offset (0.1) to avoid the annihilation effect
        result = std::pow(weight_factor, alfa) + 0.1;
        result /= (term_2 * term_1);
    }
    return result;
}

double kernel(double x, double y, int degree, double alfa)
{
    //  Compute the n-th degree Gegenbauer Mercer kernel, with special parameter alfa,
    //  defined as a product of Gegenbauer polynomials evaluated at x and y.

    double sum_result = 1.0;
    double mult_result = 1.0;

    for (int k = 1; k <= degree; k++)
    {
        if (x != 0.0 and y != 0.0)
        {
            mult_result = gegenbauerc(x, k, alfa) * gegenbauerc(y, k, alfa);
            mult_result *= weights(x, y, alfa, k);
            sum_result += mult_result;
        }
    }

    return sum_result;
}