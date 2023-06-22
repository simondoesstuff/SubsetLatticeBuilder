use arrayfire as af;

fn main() {
    println!("Hello, world!");
    println!("ArrayFire Devices: {}", af::device_count());
    // device details
    println!("{:?}", af::device_info());
}
