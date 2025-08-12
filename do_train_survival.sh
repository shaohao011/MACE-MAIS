# export CUDA_VISIBLE_DEVICES=1
# SEED=42
# SEED=46
# SEED=200
SEED=223
# renji lr: 3e-4 wd 1e-5 seed 46
# anzhen lr: 1e-4 wd 1e-5 seed 223
# tongji lr: 3e-4 wd 1e-5 seed 223
# gulou lr: 3e-4 wd 1e-5 seed 223
LOG_NAME=debug_shap_2
LOGDIR=runs/${LOG_NAME}
mkdir -p $LOGDIR
# /opt/conda/envs/py39/bin/python -u \
python -u \
main_survival.py \
--dataset ./jsons/jsons_multi_center_wCoT/anzhen.json \
--optim_lr 3e-4 \
--optim_name 'adamw' \
--lrschedule 'cosine_anneal' \
--reg_weight 1e-5 \
--batch_size=8 \
--max_epochs -1 \
--random_seed $SEED \
--num_workers 8 \
--spatial_dim 2 \
--in_channels 10 \
--out_channels 4 \
--loss_type ce \
--model_type 'conv' \
--n_folds 3 \
--fold 1 \
--logdir $LOG_NAME \
--model_index -1 \
--llm_name T5-3B \
--use_lora \
--use_text_inputs \
--pure_text_inputs \
--img_rpt_path "./img_rpt/uniformer_4_center" \
--text_inputs_keys 'Refined_Rationale_WithLabel'