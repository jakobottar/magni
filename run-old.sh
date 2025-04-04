#! /bin/bash

#SBATCH --job-name=partproto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --account=tolgalab
#SBATCH --partition=titanrtx-24
#SBATCH --output=log_%J.txt

source ~/.micromamba/etc/profile.d/mamba.sh
micromamba activate fio

export CUDA_VISIBLE_DEVICES=0

### Fold training
# for ((i = 1 ; i <= 5 ; i++ )); do 

#     echo "fold number $i"

#     ### SEM training with all 13 route classes
#     ntfy "image model" python image_trainer.py -c image.yaml -e 100 --fold-num $i --name image-model-f$i

#     ### XRD training with final material classes
#     ntfy "xrd model" python xrd_trainer.py -c xrd.yaml -e 20 --fold-num $i --name xrd-model-f$i

#     ### XRD and Image multimodal training
#     ntfy "xrd+image model" python main.py -c image.yaml -e 10 --name mm-model-f$i --checkpoint ./runs/image-model-f$i/best.pth --xrd-checkpoint ./runs/xrd-model-f$i/best.pth --fold-num $i; 
    
# done

## Single-fold training
### XRD training with final material classes
# ntfy "xrd model" python xrd_trainer.py -c xrd.yaml -e 20
# ntfy "xrd model" python xrd_trainer.py -c xrd.yaml -e 20 --name xrd-model-synxrd-2048 --join-method add

### SEM training with all route classes
# ntfy "image model" python image_trainer.py -c image.yaml -e 100 --name image-model-randinit

### XRD and Image multimodal training
# ntfy "early cat" python main.py -c image.yaml -e 10 --name confusion --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-realxrd/best.pth
# ntfy "early add" python main.py -c image.yaml -e 10 --name mm-add-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method add
# ntfy "early max" python main.py -c image.yaml -e 10 --name mm-max-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method max

# ntfy "logitmask" python main.py -c image.yaml --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-nouo3-realxrd/best.pth --join-location late --skip-train --missing-modality xrd
# ntfy "labelmask" python main.py -c image.yaml -e 10 --name mm-labelmask-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd/best.pth --use-logit-masking-baseline --skip-train

# ntfy "faketoken cat" python main.py -c image.yaml -e 10 --name mm-faketoken-cat-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd/best.pth --use-fake-token-baseline
# ntfy "faketoken add" python main.py -c image.yaml -e 10 --name mm-faketoken-add-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method add --use-fake-token-baseline
# ntfy "faketoken max" python main.py -c image.yaml -e 10 --name mm-faketoken-max-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method max --use-fake-token-baseline

# ntfy "early cat no xrd" python main.py -c image.yaml -e 10 --name mm-cat-noxrd-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd/best.pth --missing-modality xrd
# ntfy "early add no xrd" python main.py -c image.yaml -e 10 --name mm-add-noxrd-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method add --missing-modality xrd
# ntfy "early max no xrd" python main.py -c image.yaml -e 10 --name mm-max-noxrd-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method max --missing-modality xrd

# ntfy "early cat no sem" python main.py -c image.yaml -e 10 --name mm-cat-nosem-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd/best.pth --missing-modality sem
# ntfy "early add no sem" python main.py -c image.yaml -e 10 --name mm-add-nosem-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method add --missing-modality sem
# ntfy "early max no sem" python main.py -c image.yaml -e 10 --name mm-max-nosem-synxrd --checkpoint ./runs/image-model/best.pth --xrd-checkpoint ./runs/xrd-model-synxrd-2048/best.pth --join-method max --missing-modality sem

# ntfy "super early fusion" python main.py -c image.yaml --name mm-superearly-synxrd --join-location super-early -e 100
# ntfy "super early fusion" python main.py -c image.yaml --name mm-superearly-allsynxrd --join-location super-early -e 100
ntfy "early fusion" python main.py -c image.yaml --checkpoint ./image-nouo3.pth --xrd-checkpoint ./xrd-nouo3.pth --join-location early --name mm-early 
# ntfy "late fusion" python main.py -c image.yaml --checkpoint ./image-nouo3.pth --xrd-checkpoint ./xrd-nouo3.pth --join-location late --skip-train
# ntfy "labelmask" python main.py -c image.yaml --checkpoint ./image-nouo3.pth --xrd-checkpoint ./xrd-nouo3.pth --use-label-masking-baseline --skip-train --name mm-labelmask