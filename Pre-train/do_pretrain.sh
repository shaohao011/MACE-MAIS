
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--dataset \
"../jsons/jsons_final_full_info/center1.json" \
"../jsons/jsons_final_full_info/center2.json" \
"../jsons/jsons_final_full_info/center3.json" \
"../jsons/jsons_final_full_info/center4.json" \
--max_epochs=300 \
--start_epoch 0 \
--save_interval 50 \
--accumulation_steps 2 \
--lr_schedule 'cosine_anneal' \
--batch_size=3 \
--lr 3e-4 \
--mask_rate 0.8 \
--num_workers 16 \
--initial_checkpoint ../models/pre-train_weights/uniformer_small_k400_8x8_partial.pth \
--logdir "uniformer_4_center"
exit 0