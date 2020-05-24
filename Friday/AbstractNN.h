#pragma once
#include <vector>
#include <string>

class AbstractNN {
protected:


public:
	virtual std::vector<float> feedfoward(std::vector<float>&& v) = 0;

	virtual void train(std::vector<float>&& loss_vector) = 0;
};