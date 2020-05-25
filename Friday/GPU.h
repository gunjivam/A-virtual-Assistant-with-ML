#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "FErrors.h"
#include <map>
#include <unordered_set>


class GPU
{
	std::string fp;
	cl_platform_id platform_id;
	cl_device_id device_id;

	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	cl_context context;
	cl_program program;
	cl_command_queue queue;

	std::string source;


	cl_int ret = 0;
	
	const int TS = 16;
	
	std::map<std::string, cl_mem> buffers;
	std::map<std::string, cl_kernel> kernels;

	std::string ParseKernel(const std::string filepath);

public:
	GPU();

	~GPU();

	void ComputeMtx(std::string program, std::string v1, std::string mtx, std::string output, unsigned int nrow1, unsigned int ncol1, unsigned int ncol2);

	void matrixVectorMul(std::string b1, std::string mtx, std::string output, unsigned int l_v1, unsigned int l_out);

	void Activate(std::string program, std::string b1, std::string out, std::string gradients, unsigned int length, float extra_val);

	void Loss(std::string program, std::string b1, std::string b2, std::string out, std::string gradients, unsigned int length, float extra_val);
	
	void Compute2VecExe(std::string program, std::string b1, std::string b2, std::string output, unsigned int length);

	void Compute1VecExe(std::string program, std::string b1, std::string output, unsigned int length);

	cl_int CreateKernel(std::string program_name);

	cl_int CreateBuffer(std::string name, unsigned int buffer_size, cl_mem_flags flag);

	cl_int ReadBuffer(std::string buffer, float* vec, unsigned int buffer_size);

	cl_int ReleaseBuffer(std::string buffer);

	cl_int WriteBuffer(std::string buffer, const float* vec, unsigned int buffer_size);

	cl_int WriteBuffers(std::string* bffs, const float** vecs, unsigned int* buffer_sizes, unsigned int size);
};

