#include "hermite.hpp"

double hermite(double x, int n)
{
//    Compute the n-th Hermite polynomial evaluated at x using the
//    very robust 3-term recursion formula.

    // Base cases, when n is 0 and 1
    if (n == 0) return 1.0;
    else if (n == 1) return x;
    else
    {
        double result = 0.0;
        double first_value = 1.0;
        double second_value = x;

        for (int i = 1; i < n; i++)
        {
            result = x * second_value - i * first_value;
            first_value = second_value;
            second_value = result;
        }

        return result;
    }
}

double kernel(double x, double y, int degree)
{
//    Compute the n-th degree Hermite Mercer kernel defined
//    as a product of Hermite polynomials evaluated at x and y.

    double sum_result = 1.0;

    for (int k = 1; k <= degree; k++)
    {
        sum_result += hermite(x, k) * hermite(y, k);
    }

//    The kernel is normalized by a factor of 2^(2n) to avoid the explosion effect
    return sum_result / std::pow(2.0, 2.0 * degree);

}