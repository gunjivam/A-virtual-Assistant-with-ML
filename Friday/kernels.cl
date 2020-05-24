// TODO: Add OpenCL kernel code here.

void kernel add(global const float* a, global const float* b, global float* c){
				int gid = get_global_id(0); 
				c[gid] = a[gid] + b[gid]; 
}

__kernel void
matrixMul(__global float* A, 
          __global float* B, 
		  __global float* C, 
          int wA, int wB)
{
	const int row = get_local_id(0); // Local row ID (max: 16)
    const int col = get_local_id(1); // Local col ID (max: 16)
    const int globalRow = 16*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = 16*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of 16*16 elements of A and B
    __local float Asub[16][16];
    __local float Bsub[16][16];

	// init acc registers
	const int rfactor = 4;
	const int rts = 16/rfactor;

	float acc[rfactor];
	for(int w = 0; w < rfactor; w++){
		acc[w] = 0.0f;
	}


    // Loop over all tiles
    const int numTiles = wB/16;
    for (int t=0; t<numTiles; t++) {
		
		for(int w = 0; w < rfactor; w++){
			const int tiledRow = 16*t + row;
			const int tiledCol = 16*t + col;
			Asub[col + w*rts][row] = A[(tiledCol + w*rts)*wA + globalRow];
			Bsub[col + w*rts][row] = B[(globalCol + w*rts )*wB + tiledRow];
		}
        
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<16; k++) {
			for(int w = 0; w < rfactor; w++){
				acc[w] += Asub[k][row] * Bsub[col + w*rts][k];
			}
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
	for(int w = 0; w < rfactor; w++){
		C[(globalCol + w*rts)*wA + globalRow] = acc[w];
	}
    
}


__kernel void matrixVectorMul(__global float* matrixA, __global float* vectorB, __global float* resultVector,
    int width_A)
{
    int tx = get_global_id(0); 

    float value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectorB[k];
    }

    resultVector[tx] = value;
}


__kernel void sub(__global const float* a, __global const float* b, __global float* c){
				int gid = get_global_id(0); 
				c[gid] = a[gid] - b[gid]; 
}

__kernel void mult(__global const float* a, __global const float* b, __global float* c){
				int gid = get_global_id(0); 
				c[gid] = a[gid] * b[gid]; 
}

__kernel void scale(global const float* a, float val, global float* c){
				int gid = get_global_id(0); 
				c[gid] = a[gid] * val; 
}

void kernel shift(global const float* a, float val, global float* c){
				int gid = get_global_id(0); 
				c[gid] = a[gid] + val; 
}

void kernel negate(global const float* a, global float* c){
				int gid = get_global_id(0); 
				c[gid] = -a[gid]; 
}

void kernel raise(global const float* a, float val, global float* c){
				int gid = get_global_id(0); 
				c[gid] = pow(val, a[gid]); 
}

void kernel power(global const float* a, float val, global float* c){
				int gid = get_global_id(0); 
				c[gid] = pow(a[gid], val); 
}


void kernel init_vec(float val, global float* c, global float* d){
				int gid = get_global_id(0); 
				c[gid] = val; 
				d[gid] = 1;
}


//activation functions

void kernel sigmoid(global const float* a, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				float m_e = 2.718281828459045;
				val = 1 / (1 + pow(m_e, -a[gid]));
				c[gid] =  v;
				d[gid] =  v*(1-v);
}

void kernel tanh(global const float* a, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				float m_e = 2.718281828459045;
				val = (pow(m_e, a[gid]) - pow(m_e, -a[gid]))/(pow(m_e, a[gid]) + pow(m_e, -a[gid]));
				c[gid] =  v;
				d[gid] =  1 - pow(v, 2);
}


void kernel relu(global const float* a, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				if(a[gid] < 0){
					c[gid] = 0;
					d[gid] = 0;
				}else{
					c[gid] = a[gid];
					d[gid] = 1;
				}
}

void kernel leaky_relu(global const float* a, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				if(a[gid] < 0){
					c[gid] = val*a[gid];
					d[gid] = val;
				}else{
					c[gid] = a[gid];
					d[gid] = 1;
				}
}

void kernel softplus(global const float* a, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				float m_e = 2.718281828459045;
				c[gid] =  log(1+ pow(m_e, a[gid]));
				d[gid] = 1 / (1 + pow(m_e, -a[gid]));
}

void kernel softmax(global const float* a, float val, global float* c, global float* d, float val){
				int gid = get_global_id(0);
				float m_e = 2.718281828459045;
				c[gid] = 0; d[gid] = 1;
				if(val != 0){
					float v = pow(m_e, a[gid])/ val;
					c[gid] =  v;
					d[gid] = v*(1 - v);
				}
}


//Loss functions

__kernel void crossEntropy(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float val){
	int gid = get_global_id(0);
	if(yHat[gid]== 1){
		c[gid] = - log(y[gid]);
		d[gid] = 1;
		if(y[gid] != 0){
			d[gid] = - 1/y[gid];
		}
	}else{
		c[gid] = - log(1 - y[gid]);
		d[gid] = 1;
		if(y[gid] != 1){
			d[gid] = 1/(1-y[gid]);
		}
	}
}


__kernel void hinge(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float val){
	int gid = get_global_id(0);
	val = 1 - yHat[gid]*y[gid];
	c[gid] = 0; d[gid] = 1;
	if(val > 0){
		c[gid] = val; 
		d[gid] = - yHat[gid];
	}
}


__kernel void huber(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float val){
	int gid = get_global_id(0);
	val = y[gid] - yHat[gid];
	d[gid] = 1;
	if(val < 0) { val *= -1;  d[gid] = -1; }
	if(val < delta){
		c[gid] = 0.5* pow(val, 2);
		d[gid] *= val;
	}else{
		c[gid] = delta*(val - 0.5*delta);
		d[gid] *= delta*y[gid];
	}	
}

__kernel void kullback_leiber(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float delta){
	int gid = get_global_id(0);
	c[gid] = 0; d[gid] = 1;
	if(y[gid] != 0){
		c[gid] = yHat[gid] * log(yHat[gid]/y[gid]);
		d[gid] = (1/(yHat[gid]/y[gid]))*(-yHat[gid]/pow(y[gid], 2));
	}
}

__kernel void l1(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float val){
	int gid = get_global_id(0);
	val = yHat[gid] - y[gid];
	d[gid] = 1;
	if(val < 0) { val *= -1; d[gid] = -1; }
	c[gid] = val;
}

__kernel void l2(__global const float* y, __global const float* yHat, __global float* c, __global float* d, float val){
	int gid = get_global_id(0);
	c[gid] = pow(yHat[gid] - y[gid], 2);
	d[gid] = 2*(yHat[gid] - y[gid]);
}