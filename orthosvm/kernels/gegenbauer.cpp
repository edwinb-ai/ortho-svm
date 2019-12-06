/*cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']
cfg['sources'] = ['src/gegenb.cpp']
%>
*/
#include <pybind11/pybind11.h>
#include "src/gegenbauer.h"

PYBIND11_MODULE(gegenbauer, m)
{
    m.def("gegenbauer", &gegenbauer);

    m.def("kernel", &kernel);
}