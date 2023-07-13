use cudarc::driver::result::{device as res_device};
use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use cudarc::driver::result::{init as cuda_driver_init};

use bit_set::BitSet;
use std::io::BufRead;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};


/// Returns the nodes as a bit buffer, the number of nodes, and the length of each node in the buffer
fn load_data(filename: &str) -> (Vec<u64>, usize, u32) {
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
    
    // max feature needs to be a multiple of 64
    max_feature = (max_feature + 63) & !63;
   
    let mut node_buffer = Vec::new();
    let node_amnt = nodes.len();
    
    for node in nodes {
        let as_bit_vec = node.into_bit_vec();
        let mut next_chunk: u64 = 0;
        let mut remaining = 64;
        
        // skip 1 because features are 1-indexed
        for bit in as_bit_vec.iter().skip(1) {
            if remaining == 0 {
                node_buffer.push(next_chunk);
                next_chunk = 0;
                remaining = 64;
            }
            
            if bit {
                next_chunk |= 1 << (remaining - 1);
            }
            
            remaining -= 1;
        }

        node_buffer.push(next_chunk);
    }
    
    let node_len: u32 = max_feature / 64;
    return (node_buffer, node_amnt, node_len);
}


fn task_gen_thread(tasks_per_device: u64, node_amnt: usize) -> (thread::JoinHandle<()>, mpsc::Receiver<Vec<(u32, u32)>>) {
    let (tx, rx) = mpsc::channel::<Vec<(u32, u32)>>();

    let handle = thread::spawn(move || {
        let mut generator = (0..node_amnt)
                        .flat_map(|i| (i+1..node_amnt)
                        .map(move |j| (i as u32, j as u32)));
        
        'outer: loop {
            let mut next_chunk = Vec::with_capacity(tasks_per_device as usize);
            
            for _ in 0..tasks_per_device {
                if let Some(item) = generator.next() {
                    next_chunk.push(item);
                } else {
                    tx.send(next_chunk).unwrap();
                    break 'outer;
                }
            }
            
            tx.send(next_chunk).unwrap();
        }
    });

    return (handle, rx);
}


fn result_consumer_thread(node_len: u32) -> (thread::JoinHandle<()>, mpsc::Sender<Vec<u64>>) {
    let (tx, rx) = mpsc::channel::<Vec<u64>>();

    // we are currently venting results as they come in
    let handle = thread::spawn(move || {
        let mut total: usize = 0;
        
        // fn result_to_nodes(result: Vec<u64>, node_len: u64) -> Vec<BitSet> {
        //     let mut new_results = Vec::new();

        //     // each block consists of u64s of bits that correspond to features
        //     for block in result.chunks(node_len as usize) {
        //         let mut node = BitSet::with_capacity(node_len as usize * 64);
                
        //         for (chunk_index, bit_chunk) in block.iter().enumerate() {
        //             // features are 1-indexed from the left.
        //             //   eg:   0110000... is {2, 3}.
        //             for bit_index in 0..64 {
        //                 if bit_chunk & (1 << (63 - bit_index)) != 0 {
        //                     node.insert(chunk_index * 64 + bit_index + 1);
        //                 }
        //             }
        //         }
                
        //         new_results.push(node);
        //     }
            
        //     new_results
        // }
        
        // for result in &rx {
        //     let nodes = result_to_nodes(result, node_len).into_iter().filter(|node| node.len() > 0).collect::<Vec<_>>();
        //     println!("Got results {:?}", nodes);
        //     total += nodes.len();
        // }
        
        for result in &rx {
            total += result.chunks(node_len as usize).len();
            println!("Got results. {} total.", total);
        }
        
        // todo analyzing results almost certainly needs to be a thread pool
        println!("Finished consuming results. {} total.", total);
    });

    return (handle, tx);
}


fn device_thread(dev: Arc<CudaDevice>, node_buffer: Vec<u64>, node_len: u32, max_task_len: u64, result_tx: mpsc::Sender<Vec<u64>>) -> (thread::JoinHandle<()>, mpsc::Sender<Vec<(u32, u32)>>) {
    let (tx, rx) = mpsc::channel::<Vec<(u32, u32)>>();
    
    let handle = thread::spawn(move || {
        // always bind to thread before using device
        dev.bind_to_thread().unwrap();
        
        // allocate buffers
        
        let mut dev_out = unsafe { dev.alloc::<u64>((max_task_len * node_len as u64).try_into().unwrap()).expect("Failed to allocate output buffer") };
        let dev_nodes = dev.htod_copy(node_buffer).expect("Failed to copy nodes to device");
        let mut dev_op1 = unsafe { dev.alloc::<u32>(max_task_len.try_into().unwrap()).expect("Failed to allocate op1 buffer") };
        let mut dev_op2 = unsafe { dev.alloc::<u32>(max_task_len.try_into().unwrap()).expect("Failed to allocate op2 buffer") };
        // additionally:   node_len,  task_len

        // start marathon
        
        for task in rx {
            let task_len = task.len();

            // before copying, we need the new task length to match the buffer size
            let task = task.into_iter().chain(std::iter::repeat((0, 0)).take(max_task_len as usize - task_len)).collect::<Vec<_>>();

            let (host_op1, host_op2): (Vec<u32>, Vec<u32>) = task.iter().cloned().unzip();
            dev.htod_sync_copy_into(&host_op1, &mut dev_op1).expect("Failed to copy op1 to device");
            dev.htod_sync_copy_into(&host_op2, &mut dev_op2).expect("Failed to copy op2 to device");
            
            let cfg = LaunchConfig::for_num_elems(task_len as u32);
            let kernel = dev.get_func("module", "intersect_kernel").unwrap();
            
            unsafe {
                kernel
                    .launch(cfg, (
                        &mut dev_out,
                        &dev_nodes,
                        &dev_op1,
                        &dev_op2,
                        node_len,
                        task_len
                    ))
                    .expect("Failed to launch kernel");
            }
            
            println!("  --->   Launched device {}", dev.ordinal());
            
            let host_out = dev.dtoh_sync_copy(&dev_out).expect("Failed to copy output to host");
            // truncate to task length to avoid sending bad data
            let host_out = host_out.into_iter().take(task_len as usize * node_len as usize).collect::<Vec<_>>();
            result_tx.send(host_out).unwrap();
            
            println!("  <---   Reclaimed device {}", dev.ordinal());
        }
        
        drop(result_tx);
    });
    
    return (handle, tx);
}


fn main() {
    let input_file = "../data/dirty/79867.txt";

    let mem_max_in_gb = 80;
    let mem_max = mem_max_in_gb * 8*10_u64.pow(9);
    
    // intialize CUDA driver

    println!("Initializing CUDA...");
    
    cuda_driver_init().expect("Failed to initialize CUDA driver");
    let dev_count = res_device::get_count().expect("Failed to get device count") as usize;
    println!("Initialized CUDA. Detected {} devices.", dev_count);
    
    // get all devices
    
    let ptx = cudarc::nvrtc::compile_ptx(
        std::fs::read_to_string("src/intersect.cu").expect("Failed to read cu file")
    ).expect("Failed to compile PTX");
    

    println!("PTX compiled.");
    println!("Analyzing devices...");
    
    let devices = {
        let mut devices = Vec::with_capacity(dev_count);
        
        for i in 0..dev_count {
            unsafe {
                let cu_device = res_device::get(i as i32).expect("Failed to get device");
                let mem = res_device::total_mem(cu_device).expect("Failed to get device memory");
                println!("   - CUDA Dev {}: {} GB", i, mem as f64 / 10_usize.pow(9) as f64);
            }

            // using safe api
            let new_device = CudaDevice::new(i).unwrap_or_else(|e| {
                panic!("Failed to get device {}: {}", i, e);
            });

            new_device.load_ptx(ptx.clone(), "module", &["intersect_kernel"]).expect("Failed to load PTX");
            devices.push(new_device);
        }
        
        devices
    };
    
    println!("Loaded kernels into all devices.");

    // load data
    
    println!("Loading data...");
    let (node_buffer, node_amnt, node_len) = load_data(input_file);
    println!("Loaded {} nodes with (max, rounded up by 64) lengths {} bits.", node_amnt, node_len * 64);

    // calculate tasks based on ideal memory usage
    
    let max_tasks_per_device: u64 = {
        let node_len = node_len as u64;
        let node_amnt = node_amnt as u64;
        // node_amnt * node_len * 64 = node_buffer
        // max - node_buffer = remaining
        // (2 * u32) // op1, op2
        // node_len * 64 // output buffer
        // remaining / (op12 + out) = amnt of tasks
        (mem_max - node_amnt * node_len * 64)
        / (2 * 32 + node_len * 64)
    };
    
    println!("Aiming for {} GB of memory usage (per device) by providing {} tasks per batch.", mem_max_in_gb, max_tasks_per_device);

    // launch background threads
    
    let (task_thread, task_rx) = task_gen_thread(max_tasks_per_device, node_amnt);
    println!("Task generator thread ready.");
    let (result_thread, result_tx) = result_consumer_thread(node_len);
    println!("Task consumer thread ready.");
    
    // GPU worker threads
    let (device_threads, device_txs) = {
        let mut device_threads = Vec::with_capacity(dev_count);
        let mut device_txs = Vec::with_capacity(dev_count);
        
        for dev in devices {
            // unzip
            let (thread, tx) = device_thread(
                dev, // moving dev into thread
                node_buffer.clone(),
                node_len.clone(),
                max_tasks_per_device.clone(),
                result_tx.clone()
            );
            
            device_threads.push(thread);
            device_txs.push(tx);
        }
        
        (device_threads, device_txs)
    };
    
    println!("All background threads ready.");
    
    // dice and launch tasks
    
    println!("\nLaunching devices...");
    
    // timing thread
    let (timing_thread, timing_tx) = {
        let (tx, rx) = mpsc::channel::<bool>();

        let handle = thread::spawn(move || {
            let old = Instant::now();
            let mut last = old;
            
            println!("Time: 0s");
            
            loop {
                let new = Instant::now();
                
                if new - last >= Duration::from_secs(10) {
                    println!("Time: {}s", (new - old).as_secs());
                    last = new;
                }
                
                if rx.try_recv().is_ok() {
                    let new = Instant::now();
                    println!("Time: {}s", (new - old).as_secs());
                    break;
                }
            }
        });
        
        (handle, tx)
    };
    
    // iterate through devices to evenly distribute tasks
    let mut dev_txs_iter = device_txs.iter().cycle();

    for task in task_rx {
        let next_dev = dev_txs_iter.next().unwrap();
        next_dev.send(task).unwrap();
    }
    
    println!("All forseeable tasks generated and enqueued.");
    task_thread.join().unwrap();
    
    // reclaim worker threads
    
    for (thread, tx) in device_threads.into_iter().zip(device_txs.into_iter()) {
        drop(tx);
        thread.join().unwrap();
    }
    
    println!("\nDevices finished and reclaimed.");
    
    drop(result_tx);
    result_thread.join().unwrap();
    
    timing_tx.send(true).unwrap();
    drop(timing_tx);
    timing_thread.join().unwrap();
    
    println!("All systems finished.");
}