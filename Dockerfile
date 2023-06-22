FROM rust:1.67

WORKDIR /subsetlatticebuilder

COPY data ./data
COPY trickle_constr ./trickle_constr

# RUN RUSTFLAGS="-C target-cpu=native" cargo build --release
