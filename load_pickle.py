import pickle as p

filename = 'features.pkl'
with open(filename, 'rb') as f:
	x = p.load(f)

for a,b in x.items():
	print("img_id: "+ a)
	print("img_desc: "+ str(b))
	break