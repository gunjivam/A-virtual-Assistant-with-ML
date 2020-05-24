#pragma once
#include <CL/cl.h>
#include "GPU.h"

class MathEx
{
	GPU gpu;

	std::map<std::string, void (*)(std::vector<float>&&, std::vector<float>&&)> functions;

	std::map<std::string, unsigned int> tensor_keys;

	MathEx();

public:
	static MathEx& getInstance() {
		static MathEx instance;
		return instance;
	}

	void matmul(const std::vector<float>&& x, const std::vector<std::vector<float>>&& w, std::vector<float>&& y);

	void activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra);

	void loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra);

	void addNN(const std::map<std::string, std::vector<float>>&& constant_tensors, 
		const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation, std::string loss);

	MathEx(MathEx const&) = delete;
	void operator=(MathEx const&) = delete;

};
