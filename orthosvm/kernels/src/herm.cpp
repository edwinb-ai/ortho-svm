#include "hermite.hpp"

double hermite(double x, int n)
{
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
    double sum_result = 0.0;

    for (int k = 1; k <= degree; k++)
    {
        sum_result += hermite(x, k) * hermite(y, k) / std::pow(2.0, 2.0 * degree);
    }

    return sum_result;

}