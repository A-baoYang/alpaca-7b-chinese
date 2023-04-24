torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py \
    --base_model /home/jovyan/gpt/model/bigscience/bloomz-7b1-mt \
    --model_type bloom \
    --data_dir /home/jovyan/gpt/open_gpt/alpaca-7b-chinese/data/medical/medical-qa-instruction-zhtw.json \
    --output_dir /home/jovyan/gpt/open_gpt/alpaca-7b-chinese/finetuned/bloom-7b1-mt_medical-qa-instruction \
    --lora_target_modules '["query_key_value"]' \
    --micro_batch_size 1
    --cutoff_len 512