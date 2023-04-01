FROM rust:1.67

WORKDIR /subsetlatticebuilder

COPY data ./data
COPY src ./src
COPY Cargo.toml .
COPY Cargo.lock .

RUN cargo install --path .
RUN RUSTFLAGS="-C target-cpu=native" cargo build --release

RUN ln -s target/release/subsetlatticebuilder .