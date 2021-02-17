import sys
sys.path.append(r'C:\Users\pc\Desktop\고영국\개발\AI\DeepLearning\Scratch2')
import numpy as np
from common.util import preprocess

text = 'You say goodbey and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)

print(id_to_word)