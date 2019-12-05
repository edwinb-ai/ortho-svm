/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['herm.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "hermite.h"

namespace py = pybind11;

PYBIND11_MODULE(hermite, m)
{
    m.def("hermite", &hermite);
}