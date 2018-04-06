export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --wnut ../data/wnut-2016/ \
    --batch_size 50 \
    --emb w2v_twitter \
    --emb_path ../data/w2v_twitter/word2vec_twitter_model.bin \
    --save_dir ../data/exp9 \
    --num_units 200 \
    --num_filters 200 \
    --fine_tune \
    --use_char \
    --lr 0.005 \
    --decay_rate 0.05 \
    --grad_clip 5.0 \
    --oov embedding \
    --orth_word_emb_dim 200 \
    --bilstm_bilstm


# CUDA_VISIBLE_DEVICES=1 python -m pdb train.py \
#     --wnut ../data/wnut-2016/ \
#     --batch_size 5 \
#     --emb w2v_twitter \
#     --emb_path ../data/w2v_twitter/word2vec_twitter_model.bin \
#     --save_dir ../data/exp2 \
#     --num_units 200 \
#     --num_filters 200 \
#     --fine_tune \
#     --use_char \
#     --lr 0.001 \
#     --decay_rate 0.1 \
#     --grad_clip 5.0 \
#     --oov embedding \
#     --orth_emb_dim 200

# CUDA_VISIBLE_DEVICES=1 run train.py \
#     --wnut ../data/wnut-2016/ \
#     --batch_size 5 \
#     --emb w2v_twitter \
#     --emb_path ../data/w2v_twitter/word2vec_twitter_model.bin \
#     --save_dir ../data/exp2 \
#     --num_units 200 \
#     --num_filters 200 \
#     --fine_tune \
#     --use_char \
#     --lr 0.001 \
#     --decay_rate 0.1 \
#     --grad_clip 5.0 \
#     --oov embedding \
#     --orth_word_emb_dim 200
