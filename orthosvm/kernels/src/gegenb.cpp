#include "gegenbauer.hpp"

double pochhammer(double x, int n)
{
//  Compute the Pochhammer symbol (x)_n for rising factorials
//  using the Gamma function like so:
//  (x)_n = x(x+1)(x+2)...(x+n-1)

    // Special values for the Pochhammer symbol
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
//  Compute the Gegenbauer polynomials of degree `degree` and special parameter
//  alfa using the 3-term recurrence relation

//  Compute the special values, base cases
    if (degree == 0) return 1.0;
    else if (degree == 1) return 2.0 * alfa * x;
    else
    {
        double first_value = 1.0;
        double second_value = 2.0 * alfa * x;
        double result = 0.0;

        for (int i = 2; i <= degree; i++)
        {
            result = 2.0 * x * (i + alfa - 1.0) * second_value;
            result -= (i + 2.0 * alfa - 2.0) * first_value;
            result /= i;
            first_value = second_value;
            second_value = result;
        }

        return result;
    }
}

double weights(double x, double y, double alfa)
{
//  This computes the weight function (measure) for the Gegenbauer polynomial
//  with special paramter alfa

//  A value between -0.5 and 0.5 is unity
    if (-0.5 < alfa || alfa <= 0.5) return 1.0;
    if (alfa > 0.5)
    {
        double term_1 = (1.0 - (x * x)) * (1.0 - (y * y));
        // We need to add an offset (0.1) to avoid the annhilation effect
        double result = std::pow(term_1, alfa - 0.5) + 0.1;

        return result;
    }
}

double u_scale(int k, double alfa)
{
//  Use the Pochhammer symbol to re-scale the Gegenbauer polynomials of degree `k`
//  and special parameter alfa
    double term_1 = 1.0 / std::sqrt(k + 1.0); // ? k is partial or absolute degree?
    double term_2 = pochhammer(2.0 * alfa, k) / pochhammer(1.0, k);

    return term_1 * term_2;
}

double kernel(double x, double y, int degree, double alfa)
{
//  Compute the n-th degree Gegenbauer Mercer kernel, with special parameter alfa,
//  defined as a product of Gegenbauer polynomials evaluated at x and y.

    double sum_result = 0.0;
    double mult_result = 1.0;

    for (int k = 1; k <= degree; k++)
    {
        mult_result = gegenbauerc(x, k, alfa) * gegenbauerc(y, k, alfa);
        mult_result *= weights(x, y, alfa);
        mult_result *= std::pow(u_scale(k, alfa), 2.0);
        sum_result += mult_result;
    }

    return sum_result;
}