#!/bin/bash
#SBATCH --job-name=classic_ml
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1

deactivate
module purge

module load Python/3.10.8-GCCcore-12.2.0
source $HOME/venvs/ep/bin/activate
which python
python -V

start=$(date +%s)
cp -r /scratch/$USER/emoji-prediction/emoji_prediction/ $TMPDIR/ep/
end=$(date +%s)
duration=$((end - start))
echo "Copying took $duration seconds."

python $TMPDIR/ep/classic_model_api.py