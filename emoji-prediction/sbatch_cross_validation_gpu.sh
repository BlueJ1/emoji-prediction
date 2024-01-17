#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=124G
#SBATCH --cpus-per-task=5

deactivate
module purge

module load cuDNN/8.4.1.50-CUDA-11.7.0
module load Python/3.10.8-GCCcore-12.2.0
source $HOME/venvs/ep/bin/activate
which python
python -V

start=$(date +%s)
cp -r /scratch/s4776380/emoji-prediction/emoji-prediction/ $TMPDIR/ep/
end=$(date +%s)
duration=$((end - start))
echo "Copying took $duration seconds."

python $TMPDIR/ep/cross_validation.py nn test
