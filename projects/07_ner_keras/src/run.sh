CUDA_VISIBLE_DEVICES=2 python -u train.py \
    --train ../data/wnut-2016/train_dev \
    --dev ../data/wnut-2016/dev_2015 \
    --test ../data/wnut-2016/test \
    --batch_size 20 \
    --emb w2v_twitter \
    --emb_path ../data/w2v_twitter/word2vec_twitter_model.bin \
    --num_units 200 \
    --num_filter 200 \
    --fine_tune \
    --lr 0.001 \
    --decay_rate 0.01 \
    --grad_clip 5 \
    --oov random \
    --orth_word_emb_dim 200

# run train.py \
#     --train ../data/wnut-2016/train_dev \
#     --dev ../data/wnut-2016/dev_2015 \
#     --test ../data/wnut-2016/test \
#     --batch_size 20 \
#     --emb w2v_twitter \
#     --emb_path ../data/w2v_twitter/word2vec_twitter_model.bin \
#     --num_units 200 \
#     --num_filter 200 \
#     --fine_tune \
#     --lr 0.001 \
#     --decay_rate 0.01 \
#     --grad_clip 5 \
#     --oov random \
#     --orth_word_emb_dim 200

