clear
cargo build --release

if [ $? == 0 ]
then
    cargo run --release &
   # sleep 0.5
    clear
fi




