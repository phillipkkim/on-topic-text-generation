import torch
import os
import pickle
from transformers import BertModel, BertTokenizer
from csv_dataset import HighlightsDataset

def get_encodings(csv_path, pkl_path, tokenizer, model):
	dataset = HighlightsDataset(csv_path)
	pickle_out = open(pkl_path,"wb")

	# iterate through all highlights in csv
	for i in range(len(dataset)):
		print(i, '/', len(dataset))
		encoding = tokenizer.encode(dataset[i], add_special_tokens=False)
		input_ids = torch.tensor(encoding).unsqueeze(0)
		outputs = model(input_ids)
		last_hidden_states = outputs[0].squeeze(0) # The last hidden-state is the first element of the output tuple
		pickle.dump(last_hidden_states, pickle_out)

	pickle_out.close()

# implementation for batches:

# def get_encodings(path, tokenizer, model):
# 	dataset = HighlightsDataset(path)

# 	batchsize = 10

# 	# iterate through all highlights in csv
# 	for i in range(len(dataset)):
# 		highlights.append(dataset[i])

# 	result = torch.tensor([])

# 	for i in range(0, len(highlights), batchsize):
# 		print(i, '/', len(highlights))
# 		batch = highlights[i:i+batchsize]
# 		encoding = tokenizer.batch_encode_plus(batch, max_length = 512, add_special_tokens=False, pad_to_max_length = True)
# 		input_ids = torch.tensor(encoding['input_ids'])
# 		outputs = model(input_ids)
# 		last_hidden_states = outputs[0] # The last hidden-state is the first element of the output tuple
# 		result = torch.cat((result, last_hidden_states), 0)

# 	return result

def main():
	pretrained_weights = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
	model = BertModel.from_pretrained(pretrained_weights)

	train_csv_path = os.path.join(os.getcwd(), '..', 'data', 'train.csv')
	dev_csv_path = os.path.join(os.getcwd(), '..', 'data', 'dev.csv')
	test_csv_path = os.path.join(os.getcwd(), '..', 'data', 'test.csv')

	train_pkl_path = os.path.join(os.getcwd(), '..', 'data', 'bert_enc_train.pkl')
	dev_pkl_path = os.path.join(os.getcwd(), '..', 'data', 'bert_enc_dev.pkl')
	test_pkl_path = os.path.join(os.getcwd(), '..', 'data', 'bert_enc_test.pkl')

	get_encodings(train_csv_path, train_pkl_path, tokenizer, model)
	get_encodings(dev_csv_path, dev_pkl_path, tokenizer, model)
	get_encodings(test_csv_path, test_pkl_path, tokenizer, model)


if __name__ == "__main__":
    main()