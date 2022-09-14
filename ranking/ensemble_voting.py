from collections import Counter
from glob import glob

# Select language
lang = "en"
# Set path to directory with candidate files
files = glob("./*_{}_*.*".format(lang))
num_sent = 10
if lang == "es":
    num_sent = 12
cs = [[] for _ in range(num_sent)]
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        lc = 0
        for l in f:
            if not l.strip():
                continue
            sentence, cw, *candidates = l.rstrip().split("\t")
            cs[lc].extend(candidates)
            lc += 1
top10 = [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]]

