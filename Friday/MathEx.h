#pragma once
#include <CL/cl.h>
#include "GPU.h"

class MathEx
{
	GPU gpu;

	std::map<std::string, void (*)(std::vector<float>&&, std::vector<float>&&)> functions;

	MathEx();

public:
	static MathEx& getInstance() {
		static MathEx instance;
		return instance;
	}

	void setLoss(std::string loss);

	void setTensor(std::string tensor_name, const std::vector<float>&& v);

	void feedforward(std::vector<float>&& input_vect,  std::string* layout);

	void activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra);

	void loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra);

	void addNN(unsigned int id, const std::map<std::string, std::vector<float>>&& constant_tensors,
		const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation);

	MathEx(MathEx const&) = delete;
	void operator=(MathEx const&) = delete;

};
