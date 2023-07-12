use cudarc::driver::result::{device as res_device};
use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync, CudaSlice};
use cudarc::driver::result::{init as cuda_driver_init};

use std::sync::{Arc, mpsc};
use std::thread;


fn play_with_gpu(device: Arc<CudaDevice>, N: usize, default: usize) -> thread::JoinHandle<()> {
    let handle = thread::spawn(move || {
        // always bind to thread before using device
        device.bind_to_thread().expect("Failed to bind to thread");
        
        let mut dev_in = device.alloc_zeros::<usize>(N).unwrap();
        let mut dev_out = device.alloc_zeros::<usize>(N).unwrap();
        device.htod_sync_copy_into(&vec![default; N], &mut dev_in).unwrap(); // default values
        
        for i in 0..5 {
            unsafe {
                let kernel = device.get_func("module", "mult").expect("Failed to get kernel");
                let config = LaunchConfig::for_num_elems(N as u32);
    
                kernel.launch(config, (
                    &dev_out, &dev_in, N.clone()
                )).expect("Failed to launch kernel");
            }
            
            println!("Kernel launched.");
            
            let host_out = {
                let mut buffer = vec![0_usize; N];
                device.dtoh_sync_copy_into(&dev_out, &mut buffer).unwrap();
                buffer
            };
            
            println!("dev[{}] host_out[0] = {}, len: {}", device.ordinal(), host_out[0], host_out.len());
            
            // reuse out as in
            
            device.htod_sync_copy_into(&host_out, &mut dev_in).unwrap();
            
            println!("Dev {} on iter {} done.", device.ordinal(), i);
        }
    });

    return handle;
}


fn main() {
    println!("Initializing CUDA...");
    cuda_driver_init().expect("Failed to initialize CUDA driver");
    println!("CUDA initialized.");
    
    let ptx = cudarc::nvrtc::compile_ptx("extern \"C\" { typedef unsigned long long u64;
        __global__ void mult(u64* out, const u64* in, const u64 N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < N) {
                out[idx] = in[idx] * 2;
            }
        }
    }
    ").unwrap();
    
    let N: usize = {
        let n = 20;
        ( n * 8 * 10_usize.pow(9) ) / 64_usize
    };
    
    let dev_count = res_device::get_count().expect("Failed to get device count") as usize;

    let mut devices = (0..dev_count).map(|i| {
        let device = CudaDevice::new(i).expect("Failed to get device");
        device.load_ptx(ptx.clone(), "module", &["mult"]).expect("Failed to load PTX");
        device
    }).collect::<Vec<_>>();
    
    println!("Got {} devices.", devices.len());
    
    let mut handles = devices.into_iter().map(|device| {
        play_with_gpu(device, N, 0)
    }).collect::<Vec<_>>();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Done.");
}