use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() {
    let dev = CudaDevice::new(0).unwrap();
	println!("{:?}", dev);
	
	let ptx = cudarc::nvrtc::compile_ptx("
		extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const int numel) {
			unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i < numel) {
				out[i] = sin(inp[i]);
			}
		}").unwrap();	
	println!("PTX compiled");
	
    dev.load_ptx(ptx, "module", &["sin_kernel"]).unwrap();
	println!("PTX loaded");

    // and then retrieve the function with `get_func`
    let f = dev.get_func("module", "sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = dev.htod_copy(a_host.into()).unwrap();
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }.unwrap();

    let a_host_2 = dev.sync_reclaim(a_dev).unwrap();
    let b_host = dev.sync_reclaim(b_dev).unwrap();

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());
}