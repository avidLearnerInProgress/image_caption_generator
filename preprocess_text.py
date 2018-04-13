import string

#Load text containing descriptions
def load_doc(fname):
	with open(fname, 'r') as file:
		text = file.read()
	return text

#Load 5 descriptions associated with each image
def load_descriptions(doc):
	mapping = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2: #ignore images without image id and image description
			continue

		#get image id and image description	
		img_id, img_desc = tokens[0], tokens[1:]
		img_id = img_id.split('.')[0]
		img_desc = ' '.join(img_desc) #get img_desc as string back
		if img_id not in mapping:
			mapping[img_id] = list()
		mapping[img_id].append(img_desc)
	return mapping

#For each ele in array of descriptions for key 'x', replace these ele with cleaned_ele
def clean_descriptions(descriptions):
	#Don't replace anything, but remove the characters that show up in third string
	table = str.maketrans('','',string.punctuation) #map first and second args and ignore char from third args
	for k, v in descriptions.items(): #here v is list param
		for i in range(len(v)):
			desc = v[i]
			#print("Original: " + desc)
			desc = desc.split() #tokenize
			desc = [ele.lower() for ele in desc] #toLowerCase
			desc = [ele.translate(table) for ele in desc] #remove punctuation marks
			desc = [ele for ele in desc if len(ele) > 1] #remove single characters 'a', 's' etc..
			desc = [ele for ele in desc if ele.isalpha()] #ignore words having numeric characters
			v[i] = ' '.join(desc)
			#print("Cleaned: " + v[i])
			#print("\n")


#from descriptions, load each word from each array corresponding to a key, into set
def to_vocabulary(descriptions):
	collection = set() #collection of all words
	for k in descriptions.keys():
		for v in descriptions[k]:
			collection.update(v.split())
	return collection

#saves key value pairs in txt file	
def save_descriptions(descriptions, fname):
	lines = []
	for k, v_list in descriptions.items():
		for v in v_list:
			lines.append(k + ' ' + v)
	data = '\n'.join(lines)
	with open(fname, 'w') as f:
		f.write(data)

def main():

	#File with all descriptions
	filename = "Flickr8k_text/Flickr8k.token.txt"
	doc = load_doc(filename)

	#descriptions is a dict containing key as img_id and values as array of 5 descriptions
	descriptions = load_descriptions(doc)
	#k,v in descriptions -- {'3671851846_60c25269df': ['A lady holding one dog while another dog is playing in the yard .', 'A woman holding a white dog points at a brown dog in the grass .', 'A woman holds a dog while another dog stands nearby in a field .', 'A woman holds her little white dog and points to a big brown dog at the bottom of the hill .', 'A woman is standing in a green field holding a white dog and pointing at a brown dog .']}
	print("Descriptions Length: %d" % len(descriptions))

	clean_descriptions(descriptions)

	vocabulary = to_vocabulary(descriptions)
	#print(vocabulary)
	print("Vocab Length: %d" % len(vocabulary))
	
	save_descriptions(descriptions, "descriptions.txt")

main()