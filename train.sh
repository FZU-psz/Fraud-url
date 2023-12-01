export CUDA_VISIBLE_DEVICES=1
python url_classify_bert.py \
--model_name "bert-base-uncased" \
--batch_size 256 --learning_rate 2e-3 \
--num_train_epochs 30 --num_class 13 \
--checkpoint_path "../model_trained/bert-base-uncased_13_smote.tar" \
--accumulation_steps 1 \
