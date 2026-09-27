import nltk

# nltk 3.9+ 的 word_tokenize 需要 punkt_tab，旧的 punkt 已不再被使用。
# 为了兼容新旧版本，两个都下载。
for pkg in ("punkt_tab", "punkt"):
    try:
        nltk.download(pkg)
        print(f"已下载: {pkg}")
    except Exception as e:
        print(f"下载 {pkg} 失败: {e}")
