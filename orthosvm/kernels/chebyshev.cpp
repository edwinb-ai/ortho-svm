/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['src/chebysh.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "src/chebyshev.hpp"

PYBIND11_MODULE(chebyshev, m)
{
    m.def("chebyshev", &chebyshev);

    // m.def("kernel", &kernel);
}