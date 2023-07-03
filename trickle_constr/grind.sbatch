#!/bin/bash
#SBATCH -p short # Partition or queue. In this case, short!
#SBATCH --job-name=trickle_constr # Job name
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --ntasks=1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

#SBATCH --mem=8gb # Memory limit
#SBATCH --time=00:10:00 # Time limit hrs:min:sec
#SBATCH --output=/Users/siwa3657/sbatchOut/trickle_const.out
#SBATCH --err=/Users/siwa3657/sbatchOut/trickle_const.err

pwd; hostname; date
echo "You've requested $SLURM_CPUS_ON_NODE core(s)."

singularity exec --writable-tmpfs docker://rust:latest /bin/bash ./build_ancestral_net.sh ../data/dirty/79867.txt ../data/soln/79867.txt
