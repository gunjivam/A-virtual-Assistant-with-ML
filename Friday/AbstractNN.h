#pragma once
#include <vector>
#include <string>

class AbstractNN {
protected:
	static unsigned int id;

	unsigned int m_id;

public:
	virtual std::vector<float> feedfoward(std::vector<float>&& v) = 0;

	virtual void train(std::vector<float>&& loss_vector) = 0;

	unsigned int getId() { return m_id; }
};