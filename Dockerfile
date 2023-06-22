FROM rust:1.67

WORKDIR /subsetlatticebuilder

COPY data .
COPY trickle_constr .
COPY inf_ancestor .

# RUN RUSTFLAGS="-C target-cpu=native" cargo build --release
