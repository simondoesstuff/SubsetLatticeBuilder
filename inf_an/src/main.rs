use cudarc::driver::result::device::{get_count as device_count};
use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync, CudaSlice};
use cudarc::driver::result::{init as cuda_driver_init};

use bit_set::BitSet;
use std::io::BufRead;
use std::sync::{Arc, mpsc};
use std::thread;


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
   
    // todo crunch bits in 64 bit chunks 
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


fn task_gen_thread(mem_max: usize, node_len: usize, node_amnt: usize) -> mpsc::Receiver<Vec<(u32, u32)>> {
    let (tx, rx) = mpsc::channel::<Vec<(u32, u32)>>();

    thread::spawn(move || {
        let max_tasks_per_device = {
            // max - node_buffer = remaining
            // (2 * u32) // op1, op2
            // node_len // output buffer
            // remaining / (op12 + out) = amnt of tasks
            (mem_max - node_amnt * node_len)
            / (2 * 32 + node_len)
        };
        
        let mut generator = (0..node_amnt)
                        .flat_map(|i| (i+1..node_amnt)
                        .map(move |j| (i as u32, j as u32)));
        
        'outer: loop {
            let mut next_chunk = Vec::new();
            
            while next_chunk.len() < max_tasks_per_device {
                if let Some(next_pair) = generator.next() {
                    next_chunk.push(next_pair);
                } else {
                    tx.send(next_chunk).unwrap();
                    break 'outer;
                }
            }
            
            tx.send(next_chunk).unwrap();
        }
    });

    return rx;
}


fn result_consumer_thread(node_len: u32) -> mpsc::Sender<Vec<bool>> {
    let (tx, rx) = mpsc::channel::<Vec<bool>>();

    thread::spawn(move || {
        let mut total = 0;

        for result in rx {
            // we are currently venting results as they come in
            // let mut results = Vec::new();
            let mut results = 0;

            for chunk in result.chunks(node_len as usize) {
                // let mut node = BitSet::new();
                
                // for (i, bit) in chunk.iter().enumerate() {
                //     if *bit {
                //         node.insert(i + 1);
                //     }
                // }
                
                // results.push(node);
                results += 1;
            }
            
            total += results;
            print!("New Results. {} new of {} total.", results, total);
            print!("    ");
            
            // for result in results {
            //     print!(" {:?}", result);
            // }
            
            println!();
        }
    });

    return tx;
}


fn launch_device(dev: &Arc<CudaDevice>, task: Vec<(u32, u32)>, node_buffer: Vec<bool>, node_len: u32) -> CudaSlice<bool> {
    let (op1, op2): (Vec<u32>, Vec<u32>) = task.iter().cloned().unzip();

    let n = op1.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);
    
    let dev_op1 = dev.htod_copy(op1).expect("Failed to copy op1 to device");
    let dev_op2 = dev.htod_copy(op2).expect("Failed to copy op2 to device");
    let dev_nodes = dev.htod_copy(node_buffer.clone()).expect("Failed to copy nodes to device");

    let mut dev_out = dev.alloc_zeros::<bool>(n * node_len as usize).expect("Failed to allocate output buffer");
    
    // get kernel
    let kernel = dev.get_func("module", "intersect_kernel").expect("Failed to get kernel");
    
    unsafe {
        kernel
            .launch(cfg, (&mut dev_out, &dev_nodes ,&dev_op1, &dev_op2, node_len as u32, n))
            .expect("Failed to launch kernel");
    }
    
    return dev_out;
}


fn main() {
    let input_file = "../data/dirty/79867.txt";
    let mem_max = ( (2) * 8*10_usize.pow(9) ) as usize;
    
    
    // intialize CUDA driver

    println!("Initializing CUDA...");
    
    cuda_driver_init().expect("Failed to initialize CUDA driver");
    let dev_count = device_count().expect("Failed to get device count") as usize;
    println!("Initialized CUDA. Detected {} devices.", dev_count);
    
    // get all devices
    
    let devices = {
        let mut devices = Vec::with_capacity(dev_count);
        
        for i in 0..dev_count {
            let new_device = CudaDevice::new(i as usize).unwrap_or_else(|e| {
                panic!("Failed to get device {}: {}", i, e);
            });

            // todo check if device has enough memory
            devices.push(new_device);
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

    // start the marathon
    
    println!("Creating and dicing tasks in the background.");
    // Thread:    Generates tasks
    let task_rx = task_gen_thread(mem_max, node_len as usize, node_amnt as usize);
    // Thread:    Handles results
    let result_tx = result_consumer_thread(node_len);
    
    // initialize buffers and launch kernels
    
    println!("Launching kernels...\n");
    
    // repeatedly send and receive GPU workers
    let mut has_tasks = true;
    while has_tasks {
        let mut out_buffers = Vec::new();

        for dev in &devices {
            let task = match task_rx.recv() {
                Ok(task) => task,
                Err(_) => {
                    has_tasks = false;
                    break;
                },
            };
            
            let out_buffer = launch_device(dev, task, node_buffer.clone(), node_len);
            out_buffers.push(out_buffer);
            println!("-->   Launched device {}", dev.ordinal());
        }
        
        // enumerate through out_buffers in reverse order, without consuming it by popping buffers off the end
        for i in (0..out_buffers.len()).rev() {
            let dev = &devices[i];
            let out_buffer = out_buffers.pop().unwrap();
            let result = dev.sync_reclaim(out_buffer).expect("Failed to reclaim buffer");
            result_tx.send(result).unwrap();
            println!("<--   Reclaimed device {}", dev.ordinal());
        }
    } // sender should automatically be dropped when tasks are done
    
    println!("All tasks complete.");
}
    











    // let mut dev_out_buffers = Vec::new();
    
    // // allocate buffers on each device
    // for (i, dev) in devices.iter().enumerate() {
    //     // pairs is two separate buffers on the device
    //     let (op1, op2): (Vec<u32>, Vec<u32>) = pairs_chunked[i].iter().cloned().unzip();

    //     let n = op1.len();
    //     let cfg = LaunchConfig::for_num_elems(n as u32);
        
    //     let dev_op1 = dev.htod_copy(op1).expect("Failed to copy op1 to device");
    //     let dev_op2 = dev.htod_copy(op2).expect("Failed to copy op2 to device");
    //     let dev_nodes = dev.htod_copy(node_buffer.clone()).expect("Failed to copy nodes to device");

    //     // todo out buffer should be a bit buffer
    //     let dev_out = dev.alloc_zeros::<bool>(n * node_len as usize).expect("Failed to allocate output buffer");
    //     dev_out_buffers.push(dev_out);
        
    //     // get kernel
    //     let kernel = dev.get_func("module", "intersect_kernel").expect("Failed to get kernel");
        
    //     unsafe {
    //         kernel
    //             .launch(cfg, (&mut dev_out_buffers[i], &dev_nodes ,&dev_op1, &dev_op2, node_len as u32, n))
    //             .expect("Failed to launch kernel");
    //     }
        
    //     println!("Launched device {}", i);
    // }
    
    // // reclaim buffers from each device
    
    // println!();
    
    // let mut raw_results = Vec::new();
    
    // for (i, dev) in devices.iter().enumerate().rev() {
    //     let dev_out = dev_out_buffers.pop().unwrap();
    //     let out = dev.sync_reclaim(dev_out).expect("Failed to reclaim buffer");
    //     raw_results.push(out); // results is in reverse order
    
    //     println!("Reclaimed device {}", i);
    // }
    
    // // aggregate results
    // // todo wayyy too slow

    // println!("\nAggregating results...");
    
    // let mut results_aggregated = Vec::new();
    
    // for (_, bit_chunk) in raw_results.iter().enumerate().rev() {
    //     let mut results = Vec::new();
        
    //     for chunk in bit_chunk.chunks(node_len as usize) {
    //         let mut node = BitSet::new();
            
    //         for (i, bit) in chunk.iter().enumerate() {
    //             if *bit {
    //                 node.insert(i + 1);
    //             }
    //         }
            
    //         results.push(node);
    //     }
        
    //     results_aggregated.push(results);
    // }
    
    // // print results
    
    // println!("Results:");
    
    // let total_results = results_aggregated.iter().fold(0, |acc, x| acc + x.len());
    
    // for (i, results) in results_aggregated.iter().enumerate() {
    //     print!("Dev {}:", i);
        
    //     for result in results {
    //         print!(" {:?}", result);
    //     }
        
    //     println!();
    // }

    // println!("\nAccumulated {} results.", total_results);
// }