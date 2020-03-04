/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['src/herm.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "src/hermite.hpp"

PYBIND11_MODULE(hermite, m)
{
    m.def("hermite", &hermite, R"pbdoc(
        Compute the n-th Hermite polynomial evaluated at x using the
        very robust 3-term recursion formula.

        Args:
            x (double): The value where the polynomial will be evaluated.
            n (int): The degree of the Hermite polynomial.

        Returns:
            double: Evaluation of x using an `n`-th degree Hermite polynomial.
    )pbdoc");

    m.def("kernel", &kernel, R"pbdoc(
        Compute the n-th degree Hermite Mercer kernel defined
        as a product of Hermite polynomials evaluated at x and y.

        Args:
            x (double): The value where the first polynomial will be evaluated.
            y (double): The value where the second polynomial will be evaluated.
            degree (int): The degree of the Hermite polynomials.

        Returns:
            double: Computation of the Hermite Mercer kernel.
    )pbdoc");
}