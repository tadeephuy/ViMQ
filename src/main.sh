export s=100
export MODEL_DIR=ViMQ
export MODEL_DIR=$MODEL_DIR"/"$s
echo "${MODEL_DIR}"
python3 main.py --model_type vimq_model \
                --model_dir $MODEL_DIR \
                --data_dir ../data \
                --seed $s \
                --do_train \
                --do_eval \
                --train_batch_size 64 \
                --save_steps 50 \
                --logging_steps 50 \
                --num_train_epochs 3 \
                --num_iteration 5 \
                --tuning_metric f1_score \
                --gpu_id 0 \
                --iternoise 1 \
                --omega 0 \
                --threshold_iou 0.9 \
                --lamda 3