#pragma once
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

class AbstractNN {
protected:
	static unsigned int id;
	
	//{ "{matmul, x, w, y}, {add, y, b, length}, {softmax, act, y}, {l1, loss, y, yhat] } 
	static std::vector<std::vector<std::string>> graph;

	std::map<std::string, std::vector<float>*> vectors;

	std::map<std::string, std::tuple<unsigned int, unsigned int, unsigned int>> dimensions;

	unsigned int m_id;

	virtual void setUp() = 0;

	virtual void RegisterNN() = 0;

	AbstractNN();

public:

	virtual std::vector<float> feedfoward(std::vector<float>&& v) = 0;

	virtual void train(std::vector<float>&& loss_vector) = 0;

	void save(std::string filename);

	void load(std::string filename);

	unsigned int getId() { return m_id; }

	void rename_vectors(std::string* vecs, unsigned int size);
};