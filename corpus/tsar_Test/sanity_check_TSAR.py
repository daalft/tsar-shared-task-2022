import glob
for file_path in glob.glob("*.tsv"):
	with open(file_path) as input_file:
		lst = []
		print("processing",file_path)
		for i, ln in enumerate(input_file):
			cols = ln.strip().split("\t")
			sent = cols[0]
			complex = cols[1]
			# print("complex",complex)
			if len(sent.split(complex)) != 2:
				lst.append(ln)
		print("\t#sents",i, "sent w/ +1 complex word",len(lst))
		for v in lst:
			print(v.strip())
		