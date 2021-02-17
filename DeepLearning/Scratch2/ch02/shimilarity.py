# coding: utf-8
import sys
sys.path.append(r'C:\Users\pc\Desktop\고영국\개발\AI\DeepLearning\Scratch2')
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbey and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']] # "you"의 단어 벡터
c1 = C[word_to_id['i']] # "i"의 단어 벡터
print(cos_similarity(c0, c1)) # 0.70... <- 1에 가까울수록 유사도가 높다