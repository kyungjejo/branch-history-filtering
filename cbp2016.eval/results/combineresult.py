import os
import re
import csv

def main():
	#dirlist = ["100/", "99/", "95/", "90/"]
	dirlist = ["90/"]
	keylist = []
	res = dict()

	for d in dirlist:
		for filename in os.listdir("all/"+d):
			if filename == '.DS_Store':
				continue
			if filename not in res:
				res[filename] = dict()
			res[filename][d] = dict()

			with open("all/"+d+filename, "r") as file:
				data = file.read()
				data = data.replace("Edge sequence list access window overflow!", "")
				data = re.sub(' +', ' ', data).replace("\t ", "").replace("\n", "").replace(" ", ":").split(":")
				data = [x for x in data if x != '']
				#print(data)
				for i in range(0, len(data), 2):
					res[filename][d][data[i]] = data[i + 1]
					if data[i] not in keylist:
						if not keylist:
							keylist.append(data[i])
						else:
							for j in range(len(keylist)):
								if keylist[j] == data[i - 2]:
									keylist.insert(j + 1, data[i])
									break

	with open("all_combined.csv", "w") as f:
		writer = csv.writer(f)
		row = ["dir", "filename"]
		row += keylist
		writer.writerow(row)
		for file in res.keys():
			for d in res[file].keys():
				row = [file, d]
				for key in keylist:
					if key in res[file][d]:
						row.append(res[file][d][key])
					else:
						row.append("")
				writer.writerow(row)


if __name__ == "__main__":
	main()