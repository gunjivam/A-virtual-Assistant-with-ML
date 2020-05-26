#include "MathEx.h"


MathEx::MathEx() : f(matmul) {

	f2e["add"] = add;
	f2e["matmul"] = matmul;
	f2e["loss"] = loss;
	f2e["act"] = act;
	f2e["conv"] = conv;
}

MathEx::~MathEx()
{
}

void MathEx::setLoss(std::string loss)
{
	gpu.CreateKernel(loss);
}

void MathEx::setTensor(std::string tensor_name, const std::vector<float>&& v)
{
	gpu.WriteBuffer(tensor_name, &v[0], v.size());
}

/*
	{ "{matmul, x, w, y, lv_1, lv_2}, {add, y, b, length}, {act, sotfmax, y, g, length, e}, {loss, L1, y, yhat, g, length, e}} 
*/

void MathEx::feedforward(std::vector<float>&& input_vect, std::vector<float>&& output_vect, const std::vector<std::vector<std::string>>&& layout)
{
	for (std::vector<std::string> v : layout) {
		f = f2e.find(v[0])->second;
		switch (f)
		{
		case add:
			gpu.Compute2VecExe("add", v[1], v[2], v[1], std::stoi(v[3]));
			break;
		case matmul:
			gpu.matrixVectorMul(v[1], v[2], v[3], std::stoi(v[4]), std::stoi(v[5]));
			break;
		case act:
			gpu.Activate(v[1], v[2], v[2], v[3], std::stoi(v[4]), std::stod(v[5]));
			break;
		case loss:
			gpu.Loss(v[1], v[2], v[3], v[2], v[4], std::stoi(v[5]), std::stod(v[6]));
			gpu.ReadBuffer(v[2], &output_vect[0], std::stoi(v[5]));
		default:
			std::cout << "graph compilation error: unidentified symbol " + v[0] << std::endl;
			break;
		}
	}
}

void MathEx::Activate(const std::string fname, std::vector<float>&& y, std::vector<float>&& gradient, const float extra)
{
}

void MathEx::Loss(const std::string fname, std::vector<float>&& y, const std::vector<float>&& yHat, const float extra)
{
}

void MathEx::addNN(unsigned int id, const std::map<std::string, std::vector<float>>&& constant_tensors,
	const std::map<std::string, unsigned int>&& nonconstant_tensors, std::string activation, std::string loss)
{
	gpu.CreateKernel(activation); 
	gpu.CreateKernel(loss);

	// x, h, c, y tensors - one tensor created if it's not the first of its type
	if (id == 1) {
		for (auto i = nonconstant_tensors.begin(); i != nonconstant_tensors.end(); i++)
			gpu.CreateBuffer(i->first, i->second, CL_MEM_READ_WRITE);
	}
	else {
		gpu.CreateBuffer("y", nonconstant_tensors.find("y")->second, CL_MEM_READ_WRITE);
	}
	
	// w and b tensors
	for (auto i = constant_tensors.begin(); i != constant_tensors.end(); i++) {

		gpu.CreateBuffer(i->first, i->second.size(), CL_MEM_READ_WRITE);
		gpu.WriteBuffer(i->first, const_cast<float*>(&(i->second[0])), i->second.size());
	}
}

void MathEx::Train(std::vector<float>&& loss_vector, float training_rate)
{
}

