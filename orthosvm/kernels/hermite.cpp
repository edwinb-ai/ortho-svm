/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['src/herm.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "src/hermite.h"

PYBIND11_MODULE(hermite, m)
{
    m.def("hermite", &hermite);

    m.def("kernel", &kernel);
}