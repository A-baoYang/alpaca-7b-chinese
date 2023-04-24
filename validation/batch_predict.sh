CUDA_VISIBLE_DEVICES=0,1,2,3 python batch_generate.py \
    --model_name_or_path /home/jovyan/gpt/model/bigscience/bloomz-7b1-mt \
    --lora_weights ../finetuned/bloom-7b1-mt_medical-qa-instruction \
    --dev_file ../data/medical/medical-qa-instruction-zhtw-test.json \
    --dev_batch_size 2 \
    --max_length 512 \
    --output_file ../finetuned/bloom-7b1-mt_medical-qa-instruction/generate_predictions.json
