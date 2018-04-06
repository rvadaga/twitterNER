import numpy as np
import theano
import sys


embedding_path = "../0_embeddings/word2vec_twitter_model/word2vec_twitter_model.bin"


def to_unicode(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
    to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


emb_dict = dict()
f = open(embedding_path, "rb")
header = f.readline()
vocab_size, emb_size = map(int, header.split())
binary_len = np.dtype(theano.config.floatX).itemsize * emb_size
for idx in xrange(vocab_size):
    word = []
    while True:
        ch = f.read(1)
        if ch == b' ':
            break
        if ch != b'\n':  # ignore newlines in front of words
            # (some binary files have newline, some don't)
            word.append(ch)
    word = to_unicode(b''.join(word), encoding='latin-1')
    emb = np.empty([1, emb_size], dtype=theano.config.floatX)
    try: 
        emb = np.fromstring(f.read(binary_len),
                        dtype=theano.config.floatX)
    except Exception as e:
        print word, ":", str(e) 
    emb_dict[word] = emb