extern "C" {
    
    typedef unsigned long long u64;
    typedef unsigned int u32;

    __global__ void intersect_kernel(
            u64* out,
            const u64* nodes,
            const u32* op1,
            const u32* op2,
            const u32 node_len,
            const u64 amnt)
    {
        u64 id = blockIdx.x * blockDim.x + threadIdx.x;

        if (id < amnt) {
            u64 op1_index = op1[id] * node_len;
            u64 op2_index = op2[id] * node_len;
            u64 out_index = id * node_len;
            
            for (u64 i = 0; i < node_len; i++) {
                out[out_index + i] = nodes[op1_index + i] & nodes[op2_index + i];
            }
        }
    }
    
    // __global__ void intersect_kernel(int a, int b) {
    //     int c = a + b;
    // }

}