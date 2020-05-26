#include "AbstractNN.h"

AbstractNN::AbstractNN() 
{
	m_id = AbstractNN::id;
}

void AbstractNN::save(std::string filename)
{
	std::ofstream outfile(filename + ".txt");

	for (auto v : graph) {
		for (auto part : v) {
			outfile << "my text here!" << std::endl;
		}
	}
	outfile.close();
}

void AbstractNN::load(std::string filename)
{
}

void AbstractNN::rename_vectors(std::string* vecs, unsigned int size)
{
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		vecs[i] += id;
	}
}
