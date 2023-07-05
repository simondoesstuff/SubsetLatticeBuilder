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
use cudarc::driver::CudaDevice;
use cudarc::driver::result::{init as cuda_driver_init};


fn main() {
    // intialize CUDA driver
    
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
    
    for dev in devices {
        dev.load_ptx(ptx.clone(), "module", &["intersect_kernel"]).expect("Failed to load PTX");
    }

    let kernels = devices.iter().map(|dev| {
        dev.get_func("module", "intersect_kernel").expect("Failed to get kernel")
    }).collect::<Vec<_>>();

    println!("Kernels loaded.");
    
    // initialize buffers
    
    
}