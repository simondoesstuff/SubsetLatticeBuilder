// Example
// extern C __global__ void sin_kernel(float *out, const float *inp, const int numel) {
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < numel) {
//         out[i] = sin(inp[i]);
//     }
// }



extern "C" {
    
    typedef unsigned int uint;

    __global__ void intersect_kernel(
            bool* out,
            const bool* nodes,
            const uint* op1,
            const uint* op2,
            const uint node_amnt,
            const uint node_len)
    {
        uint id = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (id < node_amnt) {
            uint op1_index = op1[id] * node_len;
            uint op2_index = op2[id] * node_len;
            uint out_index = id * node_len;
            
            for (uint i = 0; i < node_len; i++) {
                out[out_index + i] = nodes[op1_index + i] && nodes[op2_index + i];
            }
        }
    }

}