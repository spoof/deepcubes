from intentclf.models import Embedder

embedder = Embedder("/mnt/disk/models/glove-hh-embeds.kv")

text = "привет как дела? У меня хорошо"
print(embedder._get_lemmitize_words(text))

embedder.get_vector("привет как дела")
