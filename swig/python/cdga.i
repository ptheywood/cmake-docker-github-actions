%begin %{
// define SWIG_PYTHON_INTERPRETER_NO_DEBUG on windows debug builds as pythonXX_d is not packaged unless built from source
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

%module(directors="1") pycdga

%{
// Include the main library header, that should subsequently make all other required (public) headers available.
#include "cdga/cdga.h"
using namespace cdga; // @todo - is this required? Ideally it shouldn't be, but swig just dumps stuff into the global namespace. 
%}


// Expand SWIG support for the standard library
%include <stl.i>
// argc/argv support
%include <argcargv.i> 

// Enable the use of argc/argv
%apply (int ARGC, char **ARGV) { (int argc, const char **) }   

%include "cdga/version.h"
%include "cdga/Demo.h"
