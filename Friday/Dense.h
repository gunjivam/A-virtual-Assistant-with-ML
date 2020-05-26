#pragma once
#include "AbstractNN.h"

class Dense : AbstractNN
{

	std::vector<float> w, y;

	void setUp();

	virtual void RegisterNN();

public:
	Dense(unsigned int in_length, unsigned int out_length, std::string activation,
		std::pair<float, float> weight_params, std::pair<float, float> bias_params);

	std::vector<float> feedfoward(std::vector<float>&& v);

	void train(std::vector<float>&& loss_vector);
};

