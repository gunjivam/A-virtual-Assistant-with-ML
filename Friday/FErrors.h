#pragma once
#include <CL/cl.h>

void CheckEx(cl_int error);

const char* getErrorString(cl_int error);