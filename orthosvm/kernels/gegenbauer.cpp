/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['src/gegenb.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "src/gegenbauer.hpp"

PYBIND11_MODULE(gegenbauer, m)
{
    m.def("pochhammer", &pochhammer, R"pbdoc(
        Compute the Pochhammer symbol (x)_n for rising factorials
        using the Gamma function like so:
        (x)_n = x(x+1)(x+2)...(x+n-1)

        Args:
            x (double): The value where the symbol will be evaluated.
            n (int): The value of the factorial to compute.

        Returns:
            double: The Pochhammer symbol n of x.
    )pbdoc");

    m.def("gegenbauerc", &gegenbauerc, R"pbdoc(
        Compute the Gegenbauer polynomials of degree `degree` and special parameter
        alfa using the 3-term recurrence relation.
        The parameter alfa controls the family that it generalizes, e.g. alfa = 0.5
        corresponds to the Legendre polynomials, whereas alfa = 1.0 reduce to the
        Chebyshev polynomials.

        Args:
            x (double): The value where the polynomial will be evaluated.
            degree (int): The degree of the polynomial.
            alfa (double): The special parameter alfa.

        Returns:
            double: Evaluation of x using an `n`-th degree alfa-Gegenbauer polynomial.
    )pbdoc");

    m.def("weights", &weights, R"pbdoc(
        This computes the weight function (measure) for the Gegenbauer polynomial
        with special parameter alfa.
        This function actually expects two parameters, `x` and `y`, so it's effectively
        a generalization of the 1-dimensional Gegenbauer polynomial.

        Args:
            x (double): The first value where the polynomial will be evaluated.
            y (double): The second value where the polynomial will be evaluated.
            alfa (double): The special parameter alfa.

        Returns:
            double: Evaluation of x and y using the weight function.
    )pbdoc");

    m.def("u_scale", &u_scale, R"pbdoc(
        Use the Pochhammer symbol to re-scale the Gegenbauer polynomials of degree `k`
        and special parameter alfa.
        This is done to prevent annhilation and explosion effects while computing the
        kernels.

        Args:
            k (int): The degree of the Gegenbauer polynomial.
            alfa (double): The special parameter alfa.

        Returns:
            double: The value that scales the given polynomial.
    )pbdoc");

    m.def("kernel", &kernel, R"pbdoc(
        Compute the n-th degree Gegenbauer Mercer kernel, with special parameter alfa,
        defined as a product of Gegenbauer polynomials evaluated at `x` and `y`.

        Args:
            x (double): The value where the first polynomial will be evaluated.
            y (double): The value where the second polynomial will be evaluated.
            degree (int): The degree of the Gegenbauer polynomials.
            alfa (double): The special parameter alfa.

        Returns:
            double: Computation of the Gegenbauer Mercer kernel.
    )pbdoc");
}