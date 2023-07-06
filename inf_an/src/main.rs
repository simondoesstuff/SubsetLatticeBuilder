// use cudarc::{
//     driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
//     nvrtc::Ptx,
// };

// fn main() {
//     let dev = CudaDevice::new(0).unwrap();
// 	println!("{:?}", dev);
	
// 	let ptx = cudarc::nvrtc::compile_ptx("
// 		extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const int numel) {
// 			unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
// 			if (i < numel) {
// 				out[i] = sin(inp[i]);
// 			}
// 		}").unwrap();	
// 	println!("PTX compiled");
	
//     dev.load_ptx(ptx, "module", &["sin_kernel"]).unwrap();
// 	println!("PTX loaded");

//     // and then retrieve the function with `get_func`
//     let f = dev.get_func("module", "sin_kernel").unwrap();

//     let a_host = [1.0, 2.0, 3.0];

//     let a_dev = dev.htod_copy(a_host.into()).unwrap();
//     let mut b_dev = a_dev.clone();

//     let n = 3;
//     let cfg = LaunchConfig::for_num_elems(n);
//     unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }.unwrap();

//     let a_host_2 = dev.sync_reclaim(a_dev).unwrap();
//     let b_host = dev.sync_reclaim(b_dev).unwrap();

//     println!("Found {:?}", b_host);
//     println!("Expected {:?}", a_host.map(f32::sin));
//     assert_eq!(&a_host, a_host_2.as_slice());
// }


use cudarc::driver::result::device::{get_count as device_count};
use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use cudarc::driver::result::{init as cuda_driver_init};

use bit_set::BitSet;
use std::io::BufRead;


/// Returns the nodes as a bit buffer, the number of nodes, and the length of each node in the buffer
fn load_data(filename: &str) -> (Vec<bool>, u32, u32) {
    // File format:    integers separated by spaces, nodes separated by newlines
    
    let file = std::fs::File::open(filename).expect("Failed to open file");
    let reader = std::io::BufReader::new(file);
    let mut nodes = Vec::new();
    
    let mut max_feature = 0;
    
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let features = line.split(" ").map(|s| s.parse::<u32>().expect("Failed to parse int")).collect::<Vec<_>>();
        let mut set = BitSet::new();
        
        for feature in features {
            set.insert(feature as usize);
            
            if feature > max_feature {
                max_feature = feature;
            }
        }
        
        nodes.push(set);
    }
    
    // bit buffer for all nodes 
    let mut node_buffer = Vec::new();
    let node_amnt = nodes.len() as u32;
    
    for node in nodes {
        let as_bit_vec = node.into_bit_vec();
        let remainder = max_feature as usize - (as_bit_vec.len() - 1);
        
        for bit in as_bit_vec.iter().skip(1) {
            node_buffer.push(bit);
        }

        node_buffer.extend(&vec![false; remainder]); // fill remaining space with 0s
    }
    
    return (node_buffer, node_amnt, max_feature);
}


fn main() {
    let input_file = "../data/dirty/3515.txt";
    
    
    // intialize CUDA driver

    println!("Initializing CUDA...");
    
    cuda_driver_init().expect("Failed to initialize CUDA driver");
    let dev_count = device_count().expect("Failed to get device count") as usize;
    println!("Initialized CUDA. Detected {} devices.", dev_count);
    
    // get all devices
    
    let devices = {
        let mut devices = Vec::with_capacity(dev_count);
        for i in 0..dev_count {
            devices.push(CudaDevice::new(i as usize).unwrap_or_else(|e| {
                panic!("Failed to get device {}: {}", i, e);
            }))
        }
        devices
    };
    
    println!("Locked on all devices.");
    
    // compile and load PTX from file
    
    let ptx = cudarc::nvrtc::compile_ptx(
        std::fs::read_to_string("src/intersect.cu").expect("Failed to read cu file")
    ).expect("Failed to compile PTX");
    
    for dev in &devices {
        dev.load_ptx(ptx.clone(), "module", &["intersect_kernel"]).expect("Failed to load PTX");
    }

    println!("Kernels loaded.");
    
    // load data
    
    println!("Loading data...");
    let (node_buffer, node_amnt, node_len) = load_data(input_file);
    println!("Loaded {} nodes with (max) lengths {}.", node_amnt, node_len);

    // create pairs of nodes to intersect
    
    println!("Creating and dicing tasks...");
    
    let mut pairs = Vec::new();
    
    for i in 0..node_amnt {
        for j in i+1..node_amnt {
            let i = i as u32;
            let j = j as u32;
            pairs.push((i, j));
        }
    }
    
    // break the pairs into chunks for each device
    let chunk_size = pairs.len() / dev_count;
    let mut pairs_chunked = Vec::new();
    
    for i in 0..dev_count {
        let start = i * chunk_size;
        let end = if i == dev_count- 1 { pairs.len() } else { (i + 1) * chunk_size };
        pairs_chunked.push(&pairs[start..end]);
    }
    
    // initialize buffers and launch kernels
    
    println!("\nLaunching kernels...");

    let mut dev_out_buffers = Vec::new();
    
    // allocate buffers on each device
    for (i, dev) in devices.iter().enumerate() {
        // pairs is two separate buffers on the device
        let (op1, op2): (Vec<u32>, Vec<u32>) = pairs_chunked[i].iter().cloned().unzip();

        let n = op1.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        
        let dev_op1 = dev.htod_copy(op1).expect("Failed to copy op1 to device");
        let dev_op2 = dev.htod_copy(op2).expect("Failed to copy op2 to device");
        let dev_nodes = dev.htod_copy(node_buffer.clone()).expect("Failed to copy nodes to device");

        // todo out buffer should be a bit buffer
        let dev_out = dev.alloc_zeros::<bool>(n * node_len as usize).expect("Failed to allocate output buffer");
        dev_out_buffers.push(dev_out);
        
        // get kernel
        let kernel = dev.get_func("module", "intersect_kernel").expect("Failed to get kernel");
        
        unsafe {
            kernel
                .launch(cfg, (&mut dev_out_buffers[i], &dev_nodes ,&dev_op1, &dev_op2, node_len as u32, n))
                .expect("Failed to launch kernel");
        }
        
        println!("Launched device {}", i);
    }
    
    // reclaim buffers from each device
    
    println!();
    
    let mut raw_results = Vec::new();
    
    for (i, dev) in devices.iter().enumerate().rev() {
        let dev_out = dev_out_buffers.pop().unwrap();
        let out = dev.sync_reclaim(dev_out).expect("Failed to reclaim buffer");
        raw_results.push(out); // results is in reverse order
    
        println!("Reclaimed device {}", i);
    }
    
    // aggregate results

    println!("\nAggregating results...");
    
    let mut results_aggregated = Vec::new();
    
    for (_, bit_chunk) in raw_results.iter().enumerate().rev() {
        let mut results = Vec::new();
        
        for chunk in bit_chunk.chunks(node_len as usize) {
            let mut node = BitSet::new();
            
            for (i, bit) in chunk.iter().enumerate() {
                if *bit {
                    node.insert(i + 1);
                }
            }
            
            results.push(node);
        }
        
        results_aggregated.push(results);
    }
    
    // print results
    
    println!("Results:");
    
    let total_results = results_aggregated.iter().fold(0, |acc, x| acc + x.len());
    
    for (i, results) in results_aggregated.iter().enumerate() {
        print!("Dev {}:", i);
        
        for result in results {
            print!(" {:?}", result);
        }
        
        println!();
    }

    println!("\nAccumulated {} results.", total_results);
}