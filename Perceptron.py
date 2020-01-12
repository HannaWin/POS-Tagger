#this script forms the baseline for a perceptron based part of speech tagger
#following steps are implemented: 
# 1) extract features for a given word in given pos-tagged file
# 2) build a nested dictionary with each tag and its occurring features and weighths (weights are feature counts)
# 3) have the model guess the POS tag for requested word, given the previously counted weights
# 4) tune the weights for an incorrectly guessed POS tag (increase correct and decrease incorrect feature weights)
# 5) repeat 3) and 4) until correct tag is predicted by model

import pickle

class Perceptron:

	grad_desc = 100 #choose an appropriate gradient descent

	def __init__(self, file_train, file_test, flags):
		self.file_test = file_test
		self.file_train = file_train
		self.flags = str(flags)

		self.weights = {}

		self.words_train = self.create_word_list(self.file_train)
		self.words_test = self.create_word_list(self.file_test)

		self.tags_train = self.create_tag_list(self.file_train)

		self.features_train = self.extract_features(self.words_train)
		self.features_test = self.extract_features(self.words_test)

		self.predicted_tags = self.perceptron_guess()
		self.correct_tags = self.create_tag_list(self.file_test)


	def create_word_list(self, file):
		"""creates list of words in given file"""
		words_list = []
		with open (file, "rt") as file:
			for line in file:
				if line.split():
					words_list.append(line.split()[0])

		return words_list


	def create_tag_list(self, file):
		"""creates list of tags in given file"""
		tag_list = []
		with open (file, "rt") as file:
			for line in file:
				if line.split():
					tag_list.append(line.split()[1])

		return tag_list


	def extract_features(self, word_list):
		"""extracts features for all words"""
		features = []
		index = 0			
		for word in word_list:
			#extracts features for each word and writes them into list
			feature_list = []
			
			if "1" in self.flags:
			#feature1: word itself
				feature_list.append("%s=%s" %("w", word))

			if "2" in self.flags:
			#feature2: following word
				if index < len(word_list)-1:
					feature_list.append("%s=%s" %("w+1", word_list[index+1]))


			if "3" in self.flags:
			#feature3: previous word
				if index >= 1:
					feature_list.append("%s=%s" %("w-1", word_list[index-1]))

			if "4" in self.flags:
			#feature4: word is uppercase and not first word of sentence
				if word[0].isupper() and word_list[index-1] not in (".", "!", "?"):
					feature_list.append("w=CAP")

			if "5" in self.flags:
			#feature5: second previous word
				if index >= 2:
					feature_list.append("%s=%s" %("w-2", word_list[index-2]))

			if "6" in self.flags:
			#feature6: second following word
				if index <= len(word_list)-3:
					feature_list.append("%s=%s" %("w+2", word_list[index+2]))

			if "7" in self.flags:
			#feature 7: word ends with -ing
				if word[-3:] == "ing":
					feature_list.append("suf=ing")
			#feature 8: word ends with -ed
				elif word[-2:] == "ed":
					feature_list.append("suf=ed")
			#feature 9: word ends on -ness
				elif word.endswith(('ness')):
					feature_list.append("suf=ness")
			#feature 10: words ends with -tion
				elif word.endswith(('tion')):
					feature_list.append("suf=tion")
			#feature 11: words ends with -tional
				elif word.endswith(('tional')):
					feature_list.append("suf=tional")

			if "8" in self.flags:
			#feature 12: word is a number
				if self.is_number(word):
					feature_list.append("w=NUM")

			#appends feature list of each word to one big feature list which will be returned
			features.append(feature_list)
			index += 1

		return features


	def features_word(self, word):
		""" searches for requested word in given file and returns a list of its features"""
		word_features = []
		indeces = []
		ind = 0

		#writes all indeces of requested word in file's file into a list
		for w in self.words_test:
			if w == word:
				feats = self.features_test[ind]
				word_features += feats
			ind += 1

		return word_features



	def initialize_weights(self):
		"""assigns weights (=counts) to each non-zero-feature
		output is a nested dictionary: {tag1: {feature1: weight1}, {feature2: weight2}} etc."""
		weights = {}
		index = 0

		for tag in self.tags_train:
			feat = self.features_train[index]
			try:
				#if tag already in dictionary, updates weights (= features' counts)
				for f in feat:
					try:
						weights[tag][f] += 1
					except KeyError:
						weights[tag][f] = 1
			#writes tags into the dictionary (first key)
			except KeyError:
				weights[tag] = {}
				#writes features for tag into dictionary (second key + value)
				for f in feat:
					weights[tag][f] = 1
			index += 1

		self.save_weights(weights)
		return weights



	def save_weights(self, weights):
		"""saves the weights to a file to enable faster processing in further steps"""
		return pickle.dump(weights, open("weights.txt", "wb"))



	def retrieve_weights(self):
		"""loads previously saved weights"""
		return pickle.load(open("weights.txt", "rb"))



	def perceptron_guess(self):
		"""calculates sum of each tag's weights and returns tags with highest sum as guess"""
		guess = ""
		sum_tag = 0
		best_tag = 0
		predictions = []

		for w in self.words_test:
			#extract features for current word
			feats = self.features_word(w)
			for tag in self.weights:
				for f in feats:
					#if word's features in tag's features, add weight to tag's sum
					if f in self.weights.get(tag):
						sum_tag += self.weights[tag][f]
				#if tag's sum better than the previous sums, tag is new model's guess
				if sum_tag > best_tag:
					best_tag = sum_tag
					guess = tag
				sum_tag = 0
			#append word's tag prediction to list
			predictions.append(guess)
			best_tag = 0

		#write words and their predicted tags into file (used to evaluate model's predictions)
		with open("prediction.txt", "wt") as file:
			for t in range(len(predictions)):
				file.write(self.words_test[t] + "\t" + predictions[t] + "\n")

		return predictions



	def compare_tags(self):
		"""compares model's predicted tags with correct tags: 
		if prediction incorrect, tunes weights"""
		for i in range(len(self.correct_tags)):
			predicted_tag = self.predicted_tags[i]
			correct_tag = self.correct_tags[i]
			#if prediction incorrect
			if correct_tag != predicted_tag:
				#get arguments to call tuning function
				features_w = self.features_word(self.words_test[i])
				#tune weights
				self.tune_weights(correct_tag, predicted_tag, features_w)

		self.perceptron_guess()



	def tune_weights(self, correct_tag, incorrect_tag, features_word):
		"""called if predction incorrect. For relevant features:
		increases weights of correct tag, decreases weights of incorrectly guessed tag """
		for feature in features_word:
			try:
				#self.weights[correct_tag][feature] += self.grad_desc #better improvement without this line
				self.weights[incorrect_tag][feature] -= self.grad_desc
			except TypeError: #if tag not in weights
				pass
			except KeyError: #if feature not in tag's features
				pass

		self.save_weights(self.weights)
		return self.weights



	def train(self, iterations):
		"""initializes weights and makes predictions about tags
		if iterations > 1: tunes weights iterations-1 times"""
		self.weights = self.initialize_weights()
		self.perceptron_guess()

		if iterations > 1:
			for i in range(iterations-1):
				self.weights = self.retrieve_weights()
				self.compare_tags()


	@staticmethod
	def is_number(n):
		try:
			float(n)
			return True
		except ValueError:
			return False


if __name__ == '__main__':
	#running code from shell
    import sys
    file_train = sys.argv[1]
    file_test = sys.argv[2]
    flags = sys.argv[3]
    exe = Perceptron(file_train, file_test, flags)
    exe.train(1)
