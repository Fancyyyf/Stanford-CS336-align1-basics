import pickle

# 必须使用 'rb' 模式（read binary），即二进制读取模式
with open('cache/cached_counts_TinyStoriesV2-GPT4-valid.txt.pkl', 'rb') as f:
    data = pickle.load(f)

print(data) # 此时你会看到原始的 Python 对象