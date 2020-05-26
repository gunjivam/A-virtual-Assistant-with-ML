#pragma once
#include <vector>
#include <string>
#include <omp.h>

class CPU
{
	CPU();
	~CPU();

	float dot(const std::vector<float>&& v1, float* v2, const int index);

public:
	void matmul(std::vector<float>&& x, std::vector<float>&& w, std::vector<float> && y);

	void add(const std::vector<float>&& v1, const std::vector<float>&& v2, std::vector<float>&& output);

	void loss(std::string name, const std::vector<float>&& y, const std::vector<float>&& yHat,
		std::vector<float>&& output, float extra);

	void activate(std::string name, const std::vector<float>&& y, std::vector<float>&& output, float extra);

	void train(const std::vector<float>&& loss_vector, float training_rate);

	static CPU& getInstance() {
		static CPU instance;
		return instance;
	}

	CPU(CPU const&) = delete;
	void operator=(CPU const&) = delete;
};

