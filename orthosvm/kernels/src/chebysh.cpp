#include "chebyshev.hpp"

double chebyshev(double x, int n)
{
//  Compute the n-th Chebyshev polynomial evaluated at x using the
//  very robust 3-term recursion formula.

    if (n == 0) return 1.0;
    else if (n == 1) return x;
    else
    {
        double first_value = 1.0;
        double second_value = x;
        double result = 0.0;

        for (int i = 2; i <= n; i++)
        {
            result = 2.0 * x * second_value - first_value;
            first_value = second_value;
            second_value = result;
        }

        return result;
    }
}

double kernel(double x, double y, int degree)
{
//  Compute the n-th degree Chebyshev Mercer kernel defined
//  as a product of Chebyshev polynomials evaluated at x and y.

    double sum_result = 1.0;

    for (int k = 1; k <= degree; k++)
    {
        if (x != 0.0 and y != 0.0)
            sum_result += chebyshev(x, k) * chebyshev(y, k);
    }

//  Avoid the explosion effect by adding a small offset (0.005)
    return sum_result / std::sqrt(1.005 - (x * y));
}