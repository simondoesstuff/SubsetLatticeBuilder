FROM rust:1.67

COPY ../data .
COPY . .

RUN RUSTFLAGS="-C target-cpu=native" cargo build --release
