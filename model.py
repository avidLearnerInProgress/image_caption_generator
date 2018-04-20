#The model we will develop will generate a caption given a photo, and the caption will be generated one word at a time. 
#The sequence of previously generated words will be provided as input. 
#Therefore, we will need a ‘first word’ to kick-off the generation process and a ‘last word‘ to signal the end of the caption.
#'startseq' and 'endseq'

import pickle as p
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model

#Load text into buffer
def load_doc(fname):
	with open(fname, 'r') as f:
		text = f.read()
	return text

#Load list of predefined img_id
def load_set(fname):
	doc = load_doc(fname)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0] #extract first part of image i.e. image_id
		dataset.append(identifier)
	return set(dataset) #remove duplicate enteries
 
#Load cleaned descriptions from descriptions.txt
#For each img_id, return list of descriptions
def load_dict_clean_descriptions(fname, dataset):
	doc = load_doc(fname)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split() #split by white space
		img_id, img_desc = tokens[0], tokens[1:]
		if img_id in dataset: #skip images which are not in dataset
			if img_id not in descriptions: 
				descriptions[img_id] = list()
				#add tokens
			desc = '<startseq> ' + ' '.join(img_desc) + ' <endseq>'
			descriptions[img_id].append(desc)
	return descriptions

#Load photo features for dataset
def load_photo_features(fname, dataset)	:

	allfeatures = p.load(open(fname, 'rb'))
	features = {}
	for k in dataset:
		features.update({k : allfeatures[k]})
	return features

#Description text is required to be encoded to numbers
#create mapping of words to unique int values

#convert dict of desc to list of strings
def to_lines(descriptions):
	all_desc = list()
	for k in descriptions.keys():
		for v in descriptions[k]:
			all_desc.append(v)
	return all_desc

#fit tokenizer given desc
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(descriptions):
	lines = to_lines(descriptions)
	return max((len(d.split()) for d in lines))

#tokenizer, max_sequence_length, descriptions, photos
#create_sequrence transfers data into input output patterns of data 
#two input arrays to model: image_features, encoded_text
#one output array to model: next encoded word
#input text encoded as integers and fed to the word embedding layer
#photo features fed directly to model
#output will be probability distribution over all words in vocabulary
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	vocab_size = len(tokenizer.word_index) + 1
	for v in desc_list:
		#print(v)
		seq = tokenizer.texts_to_sequences([v])[0]
		#print(seq)
		for i in range(1, len(seq)):
			ip_seq, op_seq = seq[:i], seq[i]
			#print(str(ip_seq) + " : " + str(op_seq))
			#pad ip sequence
			ip_seq = pad_sequences([ip_seq], maxlen = max_length)[0]
			#encode op sequence
			op_seq = to_categorical([op_seq], num_classes = vocab_size)[0]
			#print(str(ip_seq) + " : " + str(op_seq))
			X1.append(photo)
			X2.append(ip_seq)
			y.append(op_seq)
	return array(X1), array(X2), array(y)


def _create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	vocab_size = len(tokenizer.word_index) + 1
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)
#merge_model
#Papers
	#-->https://arxiv.org/abs/1703.09137
	#-->https://arxiv.org/abs/1708.02043

#-->Photo Feature Extractor. This is a 16-layer VGG model pre-trained on the ImageNet dataset. Photos have been pre-processed with the VGG model (without the output layer). Use the extracted features from features.pkl as input
#-->Sequence Processor. This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
#-->Decoder. Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.

'''
The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. 
These are processed by a Dense layer to produce a 256 element representation of the photo.

The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. 
This is followed by an LSTM layer with 256 memory units.

Both the input models produce a 256 element vector. 
Further, both input models use regularization in the form of 50% dropout. 
This is to reduce overfitting the training dataset, as this model configuration learns very fast.

The Decoder model merges the vectors from both input models using an addition operation. 
This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.
'''
def merge_model(vocab_size, max_length):
	#feature extractor model
	inputs_1 = Input(shape = (4096,))
	feature_1 = Dropout(0.5)(inputs_1)
	feature_2 = Dense(256, activation = 'relu')(feature_1)
	#seq model
	inputs_2 = Input(shape = (max_length,))
	sequence_1 = Embedding(vocab_size, 256, mask_zero = True)(inputs_2)
	sequence_2 = Dropout(0.5)(sequence_1)
	sequence_3 = LSTM(256)(sequence_2)

	#decoder model
	decoder_1 = add([feature_2, sequence_3])
	decoder_2 = Dense(256, activation = 'relu')(decoder_1)
	outputs = Dense(vocab_size, activation = 'softmax')(decoder_2)

	model = Model(inputs = [inputs_1, inputs_2],  outputs = outputs)
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

	print(model.summary())
	plot_model(model, to_file = 'model.png', show_shapes = True)
	return model

#progressive loading
#fit_generator() in keras implements progressive loading
#generator -- create and yield one batch of examples
def data_generator(descriptions, photos, tokenizer, max_length):
	#loop for ever over images
	while 1:
		for k, v_list in descriptions.items():
			#retrieve the photo feature
			photo = photos[k][0]
			ip_img, ip_seq, op_word = create_sequences(tokenizer, max_length, v_list, photo)
			yield [[ip_img, ip_seq], op_word]


#map integer prediction back to word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

#generate description using trained model
def generate_desc(model, tokenizer, image, max_len):
	#pass startseq generating one word and then calling model recursively with generated words as input
	#until endseq is reached
	ip_text = '<startseq>'
	for i in range(max_len):
		sequence = tokenizer.texts_to_sequences([ip_text])[0]
		sequence = pad_sequences([sequence], maxlen = max_len)
		next_word = model.predict([image, sequence], verbose = 0)
		next_word = argmax(next_word)
		word = word_for_id(next_word, tokenizer)
		if word is None:
			break
		ip_text += ' ' + word
		if word == '<endseq>':
			break
	return ip_text

def evaluate_model(model, descriptions, images, tokenizer, max_len):
	actual, predicted = list(), list()
	for key, desc_list in descriptions.items():
		res = generate_desc(model, tokenizer, images[key], max_len)
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(res.split())
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0)))



#calling function
def _load_model():
	filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
	train = load_set(filename)
	print("Training data length: %d" % len(train))

	train_descriptions = load_dict_clean_descriptions("model_agnostic/descriptions.txt", train)
	#print(train_descriptions)
	print("Training descriptions length: %d" % len(train))

	train_features = load_photo_features("model_agnostic/features.pkl", train)
	print("Train features length: %d" %len(train_features))

	tokenizer = create_tokenizer(train_descriptions)
	#word_index assigns unique number to each word
	vocab_size = len(tokenizer.word_index) + 1
	print("Vocab size: "+ str(vocab_size))

	max_len = max_length(train_descriptions)
	print('Max description length: %d' % (max_len))
	
	filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
	test = load_set(filename)
	print('Test data length: %d' % len(test))
	# descriptions
	test_descriptions = load_dict_clean_descriptions('model_agnostic/descriptions.txt', test)
	print('Test Descriptions length: test=%d' % len(test_descriptions))
	# photo features
	test_features = load_photo_features('model_agnostic/features.pkl', test)
	print('Test features length: test=%d' % len(test_features))
	filename = 'model_agnostic/model_3.h5'
	model = load_model(filename)
	evaluate_model(model, test_descriptions, test_features, tokenizer, max_len)


	# prepare sequences
	#X1test, X2test, ytest = _create_sequences(tokenizer, max_len, test_descriptions, test_features)
	#print(X1test.shape)
	#print(X2test.shape)

	''' __latest__
	model = merge_model(vocab_size, max_len)
	epochs = 20
	steps = len(train_descriptions)
	#print(steps)
	test_steps = len(test_descriptions)
	for i in range(epochs):
		generator = data_generator(train_descriptions, train_features, tokenizer, max_len)
		test_generator = data_generator(test_descriptions, test_features, tokenizer, max_len)
		model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1, validation_data = test_generator, validation_steps = test_steps)
		filepath = 'model_'+str(i)+'.h5'
		model.save(filepath)
	'''

	#X1train, X2train, ytrain = create_sequences(tokenizer, max_len, train_descriptions, train_features)

	'''l = []
	epochs = 20
	for i in range(epochs):
		generator = data_generator(train_descriptions, train_features, tokenizer, max_len)
		inputs, outputs = next(generator)
		#print(inputs[0].shape)
		print(inputs[0])
		#print(inputs[1].shape)
		print(inputs[1])
		l.extend(inputs)
		#print(outputs.shape)
	'''

	#X1train, X2train, ytrain = create_sequences(tokenizer, max_len, train_descriptions, train_features)
	#model = merge_model(vocab_size, max_len)
	#filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

	#checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
	#fit model
	#model.fit([inputs[0], inputs[1]], outputs, epochs = 20, verbose = 2, callbacks = [checkpoint], validation_data = ([X1test, X2test], ytest))

_load_model()

