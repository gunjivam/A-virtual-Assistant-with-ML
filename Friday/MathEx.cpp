#include "MathEx.h"


void MathEx::setLoss(std::string loss)
{
	gpu.CreateKernel(loss);
}

void MathEx::setTensor(std::string tensor_name, const std::vector<float>&& v)
{
	gpu.WriteBuffer(tensor_name, &v[0], v.size());
}

/*
	{ "{matmul, x, w, y}, {add, y, b, y}, {softmax, act, y}, {l1, loss, y, yhat] } 
*/

void MathEx::feedforward(std::vector<float>&& input_vect, std::string* layout)
{

}

void MathEx::activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra)
{
}

void MathEx::loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra)
{
}

void MathEx::addNN(unsigned int id, const std::map<std::string, std::vector<float>>&& constant_tensors,
	const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation)
{
	gpu.CreateKernel(activation); 
	std::string s_id = std::to_string(id);

	// x, h, c, y tensors - one tensor created if it's not the first of its type
	if (id == 1) {
		for (auto i = nonconstant_tensors.begin(); i != nonconstant_tensors.end(); i++)
			gpu.CreateBuffer(i->first + s_id, i->second, CL_MEM_READ_WRITE);
	}
	else {
		gpu.CreateBuffer("y" + s_id, nonconstant_tensors.find("y"+s_id)->second, CL_MEM_READ_WRITE);
	}
	
	// w and b tensors
	for (auto i = constant_tensors.begin(); i != constant_tensors.end(); i++) {

		gpu.CreateBuffer(i->first + s_id, i->second.size(), CL_MEM_READ_WRITE);
		gpu.WriteBuffer(i->first + s_id, const_cast<float*>(&(i->second[0])), i->second.size());
	}
}

