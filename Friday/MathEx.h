#pragma once
#include <CL/cl.h>
#include "GPU.h"
#include <iostream>

class MathEx
{
	GPU gpu;

	enum Functions { matmul, add, act, loss, conv } f;

	std::map<std::string, Functions> f2e;

	MathEx();

	~MathEx();

public:
	static MathEx& getInstance() {
		static MathEx instance;
		return instance;
	}

	void setLoss(std::string loss);

	void setTensor(std::string tensor_name, const std::vector<float>&& v);

	void feedforward(std::vector<float>&& input_vect, std::vector<float>&& output_vect, const std::vector<std::vector<std::string>>&& layout);

	void Activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra);

	void Loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra);

	void addNN(unsigned int id, const std::map<std::string, std::vector<float>>&& constant_tensors,
		const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation, std::string loss);

	void Train(std::vector<float>&& loss_vector, float training_rate);

	MathEx(MathEx const&) = delete;
	void operator=(MathEx const&) = delete;

};
