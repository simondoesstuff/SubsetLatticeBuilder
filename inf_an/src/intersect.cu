extern "C" {
    
    typedef unsigned long long u64;
    typedef unsigned int u32;

    __global__ void intersect_kernel(
            u64* out,
            const u64* nodes,
            const u32 node_len,
            const u32 node_amnt,
            const u32 start,
            const u32 stop
    ){
        
        u32 id = blockIdx.x * blockDim.x + threadIdx.x + start;

        if (id < stop) {
            u32 out_index = node_len * (
                (id - start)*node_amnt - ( id*(id-1) - start*(start-1) ) / 2
            );
            
            u32 op1_index = id * node_len;
            u32 op2_index = op1_index + node_len;
            
            for (u32 i = 0; i < (node_amnt - id) * node_len; i++) {
                u32 op1_i = op1_index + (i % node_len);
                out[out_index + i] = nodes[op1_i] & nodes[op2_index + i];
            }
            
            

            // u32 op1_index = id * node_len;
            
            // for (u32 i = id + 1; i < node_amnt; i++) {
            //     u32 out_index = (id + i) * node_len; // todo fix
            //     u32 op2_index = i * node_len;
                
            //     for (u32 j = 0; j < node_len; j++) {
            //         out[out_index + j] = nodes[op1_index + j] & nodes[op2_index + j];
            //     }
            // }
        }
    }
}