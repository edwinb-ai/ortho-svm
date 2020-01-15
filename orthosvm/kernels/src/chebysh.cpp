#include "chebyshev.hpp"

double chebyshev(double x, int n)
{
    if (n == 0) return 1.0;
    else if (n == 1) return x;
    else
    {
        double first_value = 1.0;
        double second_value = x;
        double result = 0.0;

        for (int i = 1; i < n; i++)
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
    double sum_result = 0.0;

    for (int k = 0; k < degree; k++)
    {
        sum_result += chebyshev(x, k) * chebyshev(y, k);
    }

    return sum_result / std::sqrt(1.005 - (x * y));
}