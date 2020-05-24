#include "MathEx.h"

MathEx::MathEx() {

}


void MathEx::matmul(const std::vector<float>&& x, const std::vector<std::vector<float>>&& w, std::vector<float>&& y)
{

}

void MathEx::activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra)
{
}

void MathEx::loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra)
{
}

void MathEx::addNN(const std::map<std::string, std::vector<float>>&& constant_tensors, 
	const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation, std::string loss)
{
	gpu.CreateKernel(activation); 
	gpu.CreateKernel(loss);
	for (auto i = nonconstant_tensors.begin(); i != nonconstant_tensors.end(); i++) {

		auto iter = tensor_keys.find(i->first);
		if (iter == tensor_keys.end()) {
			 
		}
	}
	gpu.CreateBuffer();
}

