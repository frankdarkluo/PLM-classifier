# UPSA_pro

### Finetuned PLM Classifier
CUDA_VISIBLE_DEVICES=1 python3 run_sa.py --style_weight 1 \
--direction 1-0 \
--task sentiment \
--style_mode pipeline \
--topk 50 \
--max_steps 20 \
--output_dir new/ \
--semantic_mode kw-sent \
--keyword_weight 8 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True \
--fluency_weight 1 \
--bleu_weight 1

### Non-finetuned PLM classifier
CUDA_VISIBLE_DEVICES=0 python3 run_sa.py --style_weight 3 \
--direction 1-0 \
--style_mode plm \
--task sentiment \
--class_name EleutherAI/gpt-neo-2.7B \
--topk 10 \
--max_steps 15 \
--output_dir plm/ \
--semantic_mode kw-sent \
--keyword_weight 8 \
--sent_weight 1 \
--max_len 16 \
--action all \
--keyword_pos True

## Evaluation

bash eval.sh 