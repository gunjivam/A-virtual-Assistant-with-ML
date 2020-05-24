#include "GPU.h"
#include <iostream>
#include <chrono>

void foo_r(std::vector<float>& v) {
	for(int i = 0; i < v.size(); i++)
		v[i] += i;
}

void foo_m(std::vector<float>&& v) {
	for (int i = 0; i < v.size(); i++)
		v[i] = i;
}

int main() {
	/*std::cout << "start" << std::endl;

	std::vector<float> v(100000);

	auto start = std::chrono::high_resolution_clock::now();

	foo_m(std::move(v));

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	std::cout << v[1] << std::endl;


	start = std::chrono::high_resolution_clock::now();

	foo_r(v);

	end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	std::cout << v[1];*/


	GPU gpu;
	
	const unsigned int m = 1, n = 2048, k = 1024;
	const unsigned int sz_a = m * n, sz_b = n * k, sz_c = m * k;

	float* a = new float[sz_a];
	std::fill_n(a, sz_a, 2);
	float* b = new float[sz_b];
	std::fill_n(b, sz_b, 3);

	a[0] = 1; a[1] = 4;
	b[0] = 2; b[2] = 1; b[3] = 4;

	std::cout << "creating kernel" << std::endl;
	gpu.CreateKernel("matrixMul");
	gpu.CreateKernel("add");

	std::cout << "creating buffers" << std::endl;
	gpu.CreateBuffer("x", sz_a, CL_MEM_READ_ONLY);
	gpu.CreateBuffer("x1", sz_a, CL_MEM_READ_ONLY);
	gpu.CreateBuffer("w", sz_b, CL_MEM_READ_ONLY);
	gpu.CreateBuffer("o", sz_c, CL_MEM_READ_WRITE);

	gpu.WriteBuffers(new std::string[3]{ "x", "w", "o"}, new float* [3] {a, b, a}, new unsigned int[3] {sz_a, sz_b, sz_a}, 3u);

	gpu.matrixVectorMul("x", "w", "o", sz_a, sz_c);

	float* r = new float[sz_c];

	gpu.ReadBuffer("o", r, sz_c);

	std::cout << std::endl;

	for(int i = 0; i < 10; i++) {
		std::cout << r[i] << ", ";
	}

	delete[] r;
	delete[] a;
	delete[] b;
}