#!/bin/bash

THEANO_FLAGS=device=gpu1,floatX=float32 python -u train.py \
    --fine_tune \
    --embedding w2v_twitter \
    --oov embedding \
    --update adadelta \
    --batch_size 50 \
    --num_units 200 \
    --num_filters 200 \
    --decay_rate 0.05 \
    --grad_clipping 5 \
    --regular l2 \
    --dropout \
    --train "../../data/00_wnut_2016/train_dev" \
    --dev "../../data/00_wnut_2016/dev_2015" \
    --test "../../data/00_wnut_2016/test" \
    --embedding_dict "../../data/cambridge_ltl/word2vec_twitter_model/word2vec_twitter_model.bin" \
    --patience 20 \
    --save_model $1 \
    2>&1 | tee $1/training.log
# test ${PIPESTATUS[0]} -eq 0 && ssmtp rahul.vadaga@gmail.com < ~/email_message


# run train.py --fine_tune \
#     --embedding glove \
#     --oov embedding \
#     --update adadelta \
#     --batch_size 50 \
#     --num_units 200 \
#     --num_filters 200 \
#     --decay_rate 0.05 \
#     --grad_clipping 5 \
#     --regular l2 \
#     --dropout \
#     --train "data/wnut-2016/train" \
#     --dev "data/wnut-2016/dev" \
#     --test "data/wnut-2016/dev_2015" \
#     --embedding_dict "data/glove.6B/glove.6B.100d.gz" \
#     --patience 5 \
#     --save_model "data/exp2/"

# root=rvadaga.terminal@gmail.com
# mailhub=smtp.gmail.com:465
# rewriteDomain=gmail.com
# AuthUser=rvadaga.terminal
# AuthPass=GZcf3UrrYb
# FromLineOverride=YES
# UseTLS=YES
