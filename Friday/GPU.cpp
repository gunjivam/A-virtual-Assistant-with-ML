#include "GPU.h"
#include <iostream>
#include <chrono>


GPU::GPU(){
	std::cout << " << Getting platform <<" << std::endl;
	CheckEx(clGetPlatformIDs(1, &platform_id, &ret_num_platforms));

	std::cout << " << Getting device <<" << std::endl;
	CheckEx(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
			&device_id, &ret_num_devices));

	std::cout << " << Creating context <<" << std::endl;
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	CheckEx(ret);

	std::cout << " << Creating queue <<" << std::endl;
	queue = clCreateCommandQueue(context, device_id, 0, &ret);
	
	CheckEx(ret);

	source = ParseKernel("kernels.cl");

	std::cout << " << Creating  program <<" << std::endl;

	const char* src = const_cast<char*>(source.c_str());

	program = clCreateProgramWithSource(context, 1, &(src), NULL, &ret);
	
	CheckEx(ret);

	std::cout << " << Building  program <<" << std::endl;
	CheckEx(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

	auto program_name = "matrixVectorMul";
	kernels[program_name] = clCreateKernel(program, &(program_name[0]), &ret);
	CheckEx(ret);
}

GPU::~GPU()
{
	std::cout << "i am dead" << std::endl;
	CheckEx(clFlush(queue));
	CheckEx(clFinish(queue));

	CheckEx(clReleaseCommandQueue(queue));

	for (auto it = kernels.begin(); it != kernels.end(); it++) {
		CheckEx(clReleaseKernel(it->second));
	}

	CheckEx(clReleaseProgram(program));

	for (auto it = buffers.begin(); it != buffers.end(); it++) {
		CheckEx(clReleaseMemObject(it->second));
	}

	CheckEx(clReleaseContext(context));
}

void GPU::ComputeMtx(std::string program, std::string v1, std::string mtx, std::string output, unsigned int nrow1, unsigned int ncol1, unsigned int ncol2)
{
	cl_kernel kernel = kernels[program];
	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[v1]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[mtx]));
	CheckEx(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffers[output]));
	CheckEx(clSetKernelArg(kernel, 3, sizeof(int), (void*)&ncol1));
	CheckEx(clSetKernelArg(kernel, 4, sizeof(int), (void*)&ncol2));

	const size_t local[2] = { TS, TS / 4 };
	const size_t global[2] = { (size_t)ncol2, (size_t) ncol1 / 4 };

	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
}

void GPU::matrixVectorMul(std::string v1, std::string mtx, std::string output, unsigned int l_v1, unsigned int l_out)
{
	cl_kernel kernel = kernels["matrixVectorMul"];

	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[mtx]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[v1]));
	CheckEx(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffers[output]));
	CheckEx(clSetKernelArg(kernel, 3, sizeof(int), (void*)&l_v1));

	const size_t global_work_size = l_out;

	clFinish(queue);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	clFinish(queue);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

}

void GPU::Activate(std::string program, std::string b1, std::string out, std::string gradients, unsigned int length, float extra_val)
{
	cl_kernel kernel = kernels[program];

	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[b1]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[out]));
	CheckEx(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffers[gradients]));
	CheckEx(clSetKernelArg(kernel, 3, sizeof(float), (void*)&extra_val));

	const size_t global_work_size = length;

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
}

void GPU::Loss(std::string program, std::string b1, std::string b2, std::string out, std::string gradients, unsigned int length, float extra_val)
{
	cl_kernel kernel = kernels[program];

	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[b1]));
	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[b2]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[out]));
	CheckEx(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffers[gradients]));
	CheckEx(clSetKernelArg(kernel, 3, sizeof(float), (void*)&extra_val));

	const size_t global_work_size = length;

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
}


void GPU::Compute2VecExe(std::string program, std::string b1, std::string b2, std::string output, unsigned int length)
{
	cl_kernel kernel = kernels[program];

	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[b1]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[b2]));
	CheckEx(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffers[output]));

	const size_t global_work_size = length;

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
}

void GPU::Compute1VecExe(std::string program, std::string b1, std::string output, unsigned int length)
{
	cl_kernel kernel = kernels[program];

	CheckEx(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[b1]));
	CheckEx(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[output]));

	const size_t global_work_size = length;

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
}


cl_int GPU::CreateKernel(std::string program_name)
{
	auto iter = kernels.find(program_name);
	if (iter == kernels.end) {
		kernels[program_name] = clCreateKernel(program, &(program_name[0]), &ret);
	}
	CheckEx(ret);
	return ret;
}


std::string GPU::ParseKernel(const std::string filepath)
{
		std::ifstream stream(filepath);
		std::string line;
		std::stringstream ss;

		while (std::getline(stream, line)) {
			ss << line << '\n';
		}
		return ss.str();
}

cl_int GPU::CreateBuffer(std::string name, unsigned int buffer_size, cl_mem_flags flag)
{
	auto iter = tensor_keys.find(name);

	if ( iter == tensor_keys.end) {
		buffers[name] = clCreateBuffer(context, flag, buffer_size * sizeof(float), NULL, &ret);
		tensor_keys[name + std::to_string(iter->second + 1)] = 1;
		CheckEx(ret);
		return ret;
	}

	buffers[name + std::to_string(iter->second + 1)] = clCreateBuffer(context, flag, buffer_size * sizeof(float), NULL, &ret);
	tensor_keys[name + std::to_string(iter->second + 1)] += 1;
	CheckEx(ret);
	return ret;
}

cl_int GPU::WriteBuffer(std::string buffer, float* vec, unsigned int buffer_size)
{
	CheckEx(clEnqueueWriteBuffer(queue, buffers[buffer], CL_TRUE, 0, buffer_size * sizeof(float), vec, 0, NULL, NULL));
	return ret;
}

cl_int GPU::WriteBuffers(std::string* bffs, float** vecs, unsigned int* buffer_sizes, unsigned int size)
{
	for (int i = 0; i < size; i++) {
		ret = WriteBuffer(bffs[i], vecs[i], buffer_sizes[i]);
		if (ret != CL_SUCCESS) {
			return ret;
		}
	}
	return ret;
}

std::map<std::string, unsigned int>& GPU::getTensorKeys()
{
	return tensor_keys;
}

cl_int GPU::ReadBuffer(std::string buffer, float* vec, unsigned int buffer_size)
{
	CheckEx(clEnqueueReadBuffer(queue, buffers[buffer], CL_TRUE, 0, buffer_size * sizeof(float), vec, 0, NULL, NULL));
	clFinish(queue);
	return ret;
}

cl_int GPU::ReleaseBuffer(std::string buffer)
{
	ret = clReleaseMemObject(buffers[buffer]);
	CheckEx(ret);
	return ret;
}
