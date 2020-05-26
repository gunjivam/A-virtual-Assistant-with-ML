#include "CPU.h"

CPU::~CPU()
{
}

float CPU::dot(const std::vector<float>&& v1, float* v2, const int index)
{
	float val = 0;
#pragma omp parallel for reduction(+: val)
	for (int i = 0; i < v1.size(); i++) {
		val += v1[i] * v2[index+i];
	}
	return val;
}


void CPU::matmul(std::vector<float>&& x, std::vector<float>&& w, std::vector<float>&& y)
{
	unsigned int m = x.size(), n = (int)(w.size()/m);
	
#pragma omp parallel for
	float val = 0;
	for (int i = 0; i < n; i++) {
		y[i] = dot(std::move(x), &w[0], i * m);
	}
}

void CPU::add(const std::vector<float>&& v1, const std::vector<float>&& v2, std::vector<float>&& output)
{
#pragma omp parallel for
	for (int i = 0; i < v1.size(); i++) {
		output[i] = v1[i] + v2[i];
	}
}

void CPU::loss(std::string name, const std::vector<float>&& y, const std::vector<float>&& yHat, std::vector<float>&& output, float extra)
{

}

void CPU::activate(std::string name, const std::vector<float>&& y, std::vector<float>&& output, float extra)
{

}

void CPU::train(const std::vector<float>&& loss_vector, float training_rate)
{
}
