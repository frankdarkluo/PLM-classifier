# UPSA_pro

## Sentiment

### Non-finetuned PLM classifier
CUDA_VISIBLE_DEVICES=0 python3 run_sa.py --style_weight 3 \
--direction 1-0 \
--style_mode plm \
--task sentiment \
--class_name EleutherAI/gpt-neo-2.7B \
--topk 50 \
--max_steps 15 \
--output_dir plm/ \
--semantic_mode kw-sent \
--keyword_weight 8 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True

## Formality
CUDA_VISIBLE_DEVICES=1 python3 run_sa.py --style_weight 3 \
--direction 0-1 \
--style_mode plm \
--task formality \
--class_name EleutherAI/gpt-neo-2.7B \
--topk 50 \
--max_steps 23 \
--output_dir formality/ \
--semantic_mode kw-sent \
--keyword_weight 8 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True

