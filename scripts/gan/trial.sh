#BSUB -q gpuqueue
#BSUB -J trial
#BSUB -q gpuqueue -n 72 -gpu "num=4:j_exclusive=yes:mps=yes" -R V100 -R "span[ptile=72] rusage[mem=6]"
#BSUB -W 167:59
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

module add cuda/9.2
python zinc.py
