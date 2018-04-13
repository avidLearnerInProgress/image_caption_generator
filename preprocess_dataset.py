from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

#Extract features from all images and store it as key-value pairs
def extract_features(path):
	
	model = VGG16() #load CNN model
	model.layers.pop() #pop last layer from model as its used for predictions
	model = Model(inputs = model.inputs, outputs = model.layers[-1].output) #for output we remove last layer [-1].
	print(model.summary())

	features = dict() #key-val pairs
	for name in listdir(path):
		fname = path + '/' + name
		img = load_img(fname, target_size = (224, 224))
		img = img_to_array(img) #Array of RGB Matrix
		img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
		img = preprocess_input(img)
		feature = model.predict(img, verbose = 0)
		img_id = name.split('.')[0]
		features[img_id] = feature
		print(">%s" % name)
		break
	return features


def main():
	directory = "Flickr8k_Dataset"
	extract_features(directory)	
	print("Extracted Features: %d" % len(features))
	dump(features, open("features.pkl", 'wb'))

if __name__ == '__main__':
	main()
