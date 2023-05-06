from datasets import get_dataset_split_names, load_dataset, load_dataset_builder
from google.colab import files
from matplotlib import pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import GloVe


'''
The architecture consists of 2 stacked layers of Bi-Directional LSTM units. Each layer is a Bi-LSTM, 
which means that the input sequence is processed in both forward and backward directions by a set of LSTM units. 
The output of each LSTM unit at each time step is concatenated to form a single output vector.

In a stacked Bi-LSTM, the output of the 1st layer of Bi-LSTM units is fed as input to the 2nd layer of 
Bi-LSTM units, which in turn, allows the 2nd layer to learn a more complex representation of the input 
sequence by taking into account the outputs of the 1st layer. The final output of the stacked Bi-LSTM 
architecture is the concatenation of the output vectors from the forward and backward LSTM units at the 
top layer.

By stacking multiple layers of Bi-LSTM, the model can learn increasingly abstract representations of the input 
sequence.
'''
class ELMo(nn.Module):
    '''
    The embedding layer maps each word in the vocabulary to a dense vector of the specified embedding dimension and since pre-trained word embeddings are available 
    (here, GloVe), the embedding layer weights are initialized with these embeddings.

    Bi-Directional LSTMs are used to obtain embeddings for each word in the input 
    sentence.

    The attention weights are used to weigh the embeddings obtained at each layer of 
    the ELMo network and are initialized with Xavier initialization.

    The output layer maps the final ELMo embeddings to the specified number of classes. 
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embedding_matrix):
        super(ELMo, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.s1 = nn.Parameter(torch.rand(1))
        self.s2 = nn.Parameter(torch.rand(1))
        self.s3 = nn.Parameter(torch.rand(1))
        self.gamma = nn.Parameter(torch.rand(1))
        # Pre-Trained Embeddings from GloVe -> (34, 100)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight = nn.Parameter(self.embedding.weight, requires_grad=True) 
        # Bi-LSTM Layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True) #(34, 100)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True) #(34, 100)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)
    
    '''
    Firstly, the input sentence is passed through the embedding layer to 
    obtain the embeddings for each word. 
    
    Then these embeddings are passed through the Bi-Directional LSTM layers 
    to obtain embeddings for each word at each layer of the ELMo network.
    
    The attention weights are calculated using a softmax function and these 
    very weights are used to obtain a weighted average of the embeddings 
    obtained at each layer.
    
    The final ELMo emeddings are passed through the output layer to obtain the 
    output of the model.
    '''
    def forward(self, input_sentence):
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # Embedding Layer
        embedded = self.embedding(input_sentence)
        # Bidirectional LSTM 1
        lstm1_out, _ = self.lstm1(embedded)
        lstm1_out = self.dropout(lstm1_out)
        # Bidirectional LSTM 2
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        # Fully Connected Layer
        out = self.fc(lstm2_out)
        # Elmo embedding layer
        final_embeddings = self.gamma * (self.s1 * embedded + self.s2 * lstm1_out + self.s3 * lstm2_out)
        final_embeddings = final_embeddings.to(device)
        out = out.to(device)
        return final_embeddings, out
        

'''
Class for the Sentiment Analysis task classification model
'''
class SentimentAnalysis(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SentimentAnalysis, self).__init__()
        self.fc = nn.Linear(embedding_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = out.mean(dim=1)
        out = self.sigmoid(out)
        return out


'''
Class for the Natural Language Inference task classification model
'''
class MultiNLI(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(MultiNLI, self).__init__()
        self.fc = nn.Linear(embedding_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = out.mean(dim=1)
        out = self.sigmoid(out)
        return out


'''
Map POS tag to 1st character lemmatize() accepts
'''
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J' : wordnet.ADJ, 'N' : wordnet.NOUN, 'V' : wordnet.VERB, 'R' : wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


'''
Generation of X and y sets for the SST Corpus
Steps followed :-
(1) Word-Level Tokenization
(2) Removal of Stopwords
(3) Lemmatization
'''
def generate_sst_X_y_datasets(dataset, lemmatizer, stop_words):
    X, y = [], []
    for data in dataset:
        sentence, sentence_label = data['sentence'], data['label']
        text_tokens = word_tokenize(sentence)
        tokens_without_stopwords = [word for word in text_tokens if not word in stop_words]
        sentence_without_stopwords = (' ').join(tokens_without_stopwords)
        tokens = word_tokenize(sentence_without_stopwords)
        lemmatized_sentence_list = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
        lemmatized_sentence = (' ').join(lemmatized_sentence_list)
        X.append(lemmatized_sentence)
        if sentence_label >= 0.5:
            y.append(1)
        else:
            y.append(0)
    return X, y


'''
Generation of X and y sets for the Multi_NLI Corpus
Steps followed :-
(1) Word-Level Tokenization
(2) Removal of Stopwords
(3) Lemmatization
'''
def generate_multi_nli_X_y_datasets(dataset, lemmatizer, stop_words):
    X, y = [], []
    for data in dataset:
        premise, hypothesis, label = data['premise'], data['hypothesis'], data['label']
        sentence = premise + ' ' + hypothesis
        text_tokens = word_tokenize(sentence)
        tokens_without_stopwords = [word for word in text_tokens if not word in stop_words]
        sentence_without_stopwords = (' ').join(tokens_without_stopwords)
        tokens = word_tokenize(sentence_without_stopwords)
        lemmatized_sentence_list = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
        lemmatized_sentence = (' ').join(lemmatized_sentence_list)
        X.append(lemmatized_sentence)
        y.append(label)
    return X, y


'''
Word-to-Index Mapping - for the train dataset
'''
def create_trainset_mappings(dataset):
    vocabulary = {}
    vocabulary['<PAD>'] = 0
    vocabulary['<UNK>'] = 1
    index = 2
    for sentence in dataset:
        for word in sentence.split():
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary


'''
Creating corresponding mappings for validation and test data on the basis of the mappings 
created for the train dataset
'''
def create_mapped_dataset(train_vocab, dataset):
    mapped_dataset = []
    max_sentence_len = 0
    for sentence in dataset:
        mapped_sentence = []
        for word in sentence.split():
            if word not in train_vocab:
                mapped_sentence.append(train_vocab['<UNK>'])
            else:
                mapped_sentence.append(train_vocab[word])
        mapped_dataset.append(mapped_sentence)
        max_sentence_len = max(max_sentence_len, len(mapped_sentence))
    return mapped_dataset, max_sentence_len


'''
Padding for each of the sentences in the train, validation and test datasets
'''
def create_padded_dataset(dataset, max_len):
    padded_dataset = []
    for data in dataset:
        temp_list = data
        while len(temp_list) < max_len:
            temp_list.insert(0, 0)
        padded_dataset.append(temp_list) 
    return padded_dataset


'''
Creating the pre-trained embeddings matrix
'''
def generate_pretrained_embedding_matrix(vocab, glove, embedding_dim):
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in glove.stoi:
            embedding_matrix[index] = glove.vectors[glove.stoi[word]]
    return embedding_matrix.detach().clone()


'''
Creating ELMo embeddings for each of the mapped sentences present in the train, 
validation and test datasets
'''
def generate_elmo_embeddings(data, vocab_size, device, type, embedding_matrix):
    elmo_embeddings = []
    embedding_dim = 100
    hidden_dim = 50
    num_layers = 2
    dropout = 0.2
    epochs = 10
    learning_rate = 1e-3
    elmo = ELMo(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embedding_matrix)
    elmo = elmo.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(elmo.parameters(), lr=learning_rate)
    for i, sentence in enumerate(data):
        elmo_input = torch.tensor(sentence, dtype=torch.long).to(device)
        embeddings = None
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings, output = elmo(elmo_input)
            output = output.view(-1, vocab_size)
            loss = criterion(output, elmo_input)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print ('Epoch :', epoch, 'Index :', i, 'Loss :', loss.item())
        elmo_embeddings.append(embeddings)
    if type == 'train':
        torch.save(elmo.state_dict(), 'elmo_embeddings.pt')
    return elmo_embeddings


'''
Function for training the model for the Sentiment Analysis classifier
'''
def train_model_sst(device, model, train_loader, optimizer, loss_fn, y_train, batch_size):
    epochs = 10
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        for batch_idx, batch in enumerate(train_loader):
            input = batch[0].to(device)
            input = input.detach()
            target = batch[1].to(device)
            optimizer.zero_grad()
            out = model(input)
            y_pred = torch.argmax(out, dim=1).to(device)
            target1 = []
            for i, j in enumerate(target):
                if j == 0:
                    if y_pred[i] == 0:
                        correct += 1
                    target1.append(torch.tensor([1.0, 0.0]))
                else:
                    if y_pred[i] == 1:
                        correct += 1
                    target1.append(torch.tensor([0.0, 1.0]))
            target1 = torch.stack(target1).to(device)
            loss = loss_fn(out, target1)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss / batch_size)
        print ('Epoch :', epoch, ', Train Data Accuracy :', (correct / len(y_train)))
    plt.plot(train_losses)


'''
Function for training the model for the Sentiment Analysis classifier
'''
def train_model_multi_nli(device, model, train_loader, optimizer, loss_fn, y_train, batch_size):
    epochs = 10
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        for batch_idx, batch in enumerate(train_loader):
            input = batch[0].to(device)
            input = input.detach()
            target = batch[1].to(device)
            optimizer.zero_grad()
            out = model(input)
            y_pred = torch.argmax(out, dim=1).to(device)
            target1 = []
            for i, j in enumerate(target):
                if j == 0:
                    if y_pred[i] == 0:
                        correct += 1
                    target1.append(torch.tensor([1.0, 0.0, 0.0]))
                elif j == 1:
                    if y_pred[i] == 1:
                        correct += 1
                    target1.append(torch.tensor([0.0, 1.0, 0.0]))
                else:
                    if y_pred[i] == 2:
                        correct += 1
                    target1.append(torch.tensor([0.0, 0.0, 1.0]))
            target1 = torch.stack(target1).to(device)
            loss = loss_fn(out, target1)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss / batch_size)
        print ('Epoch :', epoch, ', Train Data Accuracy :', (correct / len(y_train)))
    plt.plot(train_losses)


'''
Function for computing the accuracies observed in validation / test datasets for the Sentiment Analysis 
classifier
'''
def sst_model_eval(device, model, data_loader, optimizer, loss_fn, y_actual, type):
    predictions = []
    num_correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input = batch[0].to(device)
            input = input.detach()
            target = batch[1].to(device)
            out = model(input)
            y_pred = torch.argmax(out, dim=1).to(device)
            for i, j in enumerate(target):
                if j == 0:
                    if y_pred[i] == 0:
                        num_correct += 1
                else:
                    if y_pred[i] == 1:
                        num_correct += 1
                predictions.append(y_pred[i])
    print (type, 'Accuracy :', (num_correct / len(y_actual)))
    return predictions


'''
Function for computing the accuracies observed in validation / test datasets for the Multi NLI  
classifier
'''
def multi_nli_model_eval(device, model, data_loader, optimizer, loss_fn, y_actual, type):
    predictions = []
    num_correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input = batch[0].to(device)
            input = input.detach()
            target = batch[1].to(device)
            out = model(input)
            y_pred = torch.argmax(out, dim=1).to(device)
            for i, j in enumerate(target):
                if j == 0:
                    if y_pred[i] == 0:
                        num_correct += 1
                elif j == 1:
                    if y_pred[i] == 1:
                        num_correct += 1
                else:
                    if y_pred[i] == 2:
                        num_correct += 1
                predictions.append(y_pred[i])
    print (type, 'Accuracy :', (num_correct / len(y_actual)))
    return predictions


'''
Utility function for the task of Sentiment Analysis on the Stanford Sentiment TreeBank corpus
'''
def sst_utils(device, lemmatizer, stop_words, glove):
    ds_builder = load_dataset_builder('sst')
    train_dataset = load_dataset('sst', split='train')
    validation_dataset = load_dataset('sst', split='validation')
    test_dataset = load_dataset('sst', split='test')
    X_train, y_train = generate_sst_X_y_datasets(train_dataset, lemmatizer, stop_words)
    X_validation, y_validation = generate_sst_X_y_datasets(validation_dataset, lemmatizer, stop_words)
    X_test, y_test = generate_sst_X_y_datasets(test_dataset, lemmatizer, stop_words)
    train_vocabulary = create_trainset_mappings(X_train)
    """
    train_mapped_dataset, max_train_sentence_len = create_mapped_dataset(train_vocabulary, X_train) 
    validation_mapped_dataset, max_validation_sentence_len = create_mapped_dataset(train_vocabulary, X_validation)
    test_mapped_dataset, max_test_sentence_len = create_mapped_dataset(train_vocabulary, X_test)
    max_sentence_len = max(max_train_sentence_len, max(max_validation_sentence_len, max_test_sentence_len))
    normal_train_data = create_padded_dataset(train_mapped_dataset, max_sentence_len)
    normal_validation_data = create_padded_dataset(validation_mapped_dataset, max_sentence_len)
    normal_test_data = create_padded_dataset(test_mapped_dataset, max_sentence_len)
    embedding_matrix = generate_pretrained_embedding_matrix(train_vocabulary, glove, 100)
    vocab_size = len(train_vocabulary)
    train_elmo_embeddings = generate_elmo_embeddings(normal_train_data, vocab_size, device, 'train', embedding_matrix)
    validation_elmo_embeddings = generate_elmo_embeddings(normal_validation_data, vocab_size, device, \
                                                          'validation', embedding_matrix)
    test_elmo_embeddings = generate_elmo_embeddings(normal_test_data, vocab_size, device, 'test', \
                                                    embedding_matrix)
    with open('train_elmo_embeddings.data', 'wb') as f1:
        pickle.dump(train_elmo_embeddings, f1)
    with open('validation_elmo_embeddings.data', 'wb') as f2:
        pickle.dump(validation_elmo_embeddings, f2)
    with open('test_elmo_embeddings.data', 'wb') as f3:
        pickle.dump(test_elmo_embeddings, f3)
    files.download('train_elmo_embeddings.data')
    files.download('validation_elmo_embeddings.data')
    files.download('test_elmo_embeddings.data')
    files.download('elmo_embeddings.pt')
    """
    file1 = open('train_elmo_embeddings.data', 'rb')
    train_elmo_embeddings = pickle.load(file1)
    file1.close()
    file2 = open('validation_elmo_embeddings.data', 'rb')
    validation_elmo_embeddings = pickle.load(file2)
    file2.close()
    file3 = open('test_elmo_embeddings.data', 'rb')
    test_elmo_embeddings = pickle.load(file3)
    file3.close()
    # Creating DataLoader objects for the Sentiment Analysis classification model
    train_data = TensorDataset(torch.stack(train_elmo_embeddings), torch.tensor(y_train))
    validation_data = TensorDataset(torch.stack(validation_elmo_embeddings), torch.tensor(y_validation))
    test_data = TensorDataset(torch.stack(test_elmo_embeddings), torch.tensor(y_test))
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    sentiment_analysis_model = SentimentAnalysis(len(train_vocabulary), 100)
    sentiment_analysis_model = sentiment_analysis_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(sentiment_analysis_model.parameters(), lr=1e-3)
    train_model_sst(device, sentiment_analysis_model, train_loader, optimizer, loss_fn, y_train, batch_size)
    validation_predictions = sst_model_eval(device, sentiment_analysis_model, validation_loader, optimizer, \
                                            loss_fn, y_validation, 'validation')
    test_predictions = sst_model_eval(device, sentiment_analysis_model, test_loader, optimizer, loss_fn, \
                                      y_test, 'test')
    print ('Validation Data ROC AUC Score :-')
    print (roc_auc_score(torch.tensor(validation_predictions).cpu().data.numpy(), \
                         torch.tensor(y_validation).cpu().data.numpy()))
    print ('Validation Data Confusion Matrix :-')
    print (confusion_matrix(torch.tensor(validation_predictions).cpu().data.numpy(), \
                            torch.tensor(y_validation).cpu().data.numpy()))
    print ('Test Data ROC AUC Score :-')
    print (roc_auc_score(torch.tensor(test_predictions).cpu().data.numpy(), torch.tensor(y_test).cpu().data.numpy()))
    print ('Test Data ROC AUC Score :-')
    print (confusion_matrix(torch.tensor(test_predictions).cpu().data.numpy(), torch.tensor(y_test).cpu().data.numpy()))


'''
Utility function for the task of Natural Language Inference on the Multi-Genre Natural Language Inference corpus
'''
def multi_nli_utils(device, lemmatizer, stop_words, glove):
    ds_builder = load_dataset_builder('multi_nli')
    train_dataset = load_dataset('multi_nli', split='train[:5%]')
    validation_matched_dataset = load_dataset('multi_nli', split='validation_matched[:50%]')
    validation_mismatched_dataset = load_dataset('multi_nli', split='validation_mismatched[:50%]')
    X_train, y_train = generate_multi_nli_X_y_datasets(train_dataset)
    X_validation_matched, y_validation_matched = generate_multi_nli_X_y_datasets(validation_matched_dataset)
    X_validation_mismatched, y_validation_mismatched = generate_multi_nli_X_y_datasets(validation_mismatched_dataset)
    train_vocabulary = create_trainset_mappings(X_train)
    """
    train_mapped_dataset, max_train_sentence_len = create_mapped_dataset(train_vocabulary, X_train) 
    validation_matched_mapped_dataset, max_validation_matched_sentence_len = \
                                                    create_mapped_dataset(train_vocabulary, X_validation_matched)
    validation_mismatched_mapped_dataset, max_validation_mismatched_sentence_len = \
                                                    create_mapped_dataset(train_vocabulary, X_validation_mismatched)
    max_sentence_len = max(max_train_sentence_len, max(max_validation_matched_sentence_len, \
                                                       max_validation_mismatched_sentence_len))
    normal_train_data = create_padded_dataset(train_mapped_dataset, max_sentence_len)
    normal_validation_matched_data = create_padded_dataset(validation_matched_mapped_dataset, max_sentence_len)
    normal_validation_mismatched_data = create_padded_dataset(validation_mismatched_mapped_dataset, max_sentence_len)
    embedding_matrix = generate_pretrained_embedding_matrix(train_vocabulary, glove, 100)
    vocab_size = len(train_vocabulary)
    train_elmo_embeddings = generate_elmo_embeddings(normal_train_data, vocab_size, device, 'train', embedding_matrix)
    validation_matched_elmo_embeddings = generate_elmo_embeddings(normal_validation_matched_data, vocab_size, \
                                                                  device, 'validation_matched')
    validation_mismatched_elmo_embeddings = generate_elmo_embeddings(normal_validation_mismatched_data, vocab_size, \
                                                                     device, 'validation_mismatched')
    with open('train_elmo_embeddings.data', 'wb') as f1:
        pickle.dump(train_elmo_embeddings, f1)
    with open('validation_matched_elmo_embeddings.data', 'wb') as f2:
        pickle.dump(validation_matched_elmo_embeddings, f2)
    with open('validation_mismatched_elmo_embeddings.data', 'wb') as f3:
        pickle.dump(validation_mismatched_elmo_embeddings, f3)
    files.download('train_elmo_embeddings.data')
    files.download('validation_elmo_embeddings.data')
    files.download('test_elmo_embeddings.data')
    files.download('elmo_embeddings.pt')
    """
    file1 = open('train_elmo_embeddings.data', 'rb')
    train_elmo_embeddings = pickle.load(file1)
    file1.close()
    file2 = open('validation_matched_elmo_embeddings.data', 'rb')
    validation_matched_elmo_embeddings = pickle.load(file2)
    file2.close()
    file3 = open('validation_mismatched_elmo_embeddings.data', 'rb')
    validation_mismatched_elmo_embeddings = pickle.load(file3)
    file3.close()
    # Creating DataLoader objects for the Sentiment Analysis classification model
    train_data = TensorDataset(torch.stack(train_elmo_embeddings), torch.tensor(y_train))
    validation_matched_data = TensorDataset(torch.stack(validation_matched_elmo_embeddings), \
                                            torch.tensor(y_validation_matched))
    validation_mismatched_data = TensorDataset(torch.stack(validation_mismatched_elmo_embeddings), \
                                               torch.tensor(y_validation_mismatched))
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_matched_loader = DataLoader(validation_matched_data, batch_size=batch_size, shuffle=True)
    validation_mismatched_loader = DataLoader(validation_mismatched_data, batch_size=batch_size, shuffle=True)
    multi_nli_model = MultiNLI(len(train_vocabulary), 100)
    sentiment_analysis_model = multi_nli_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(multi_nli_model.parameters(), lr=1e-3)
    train_model_multi_nli(device, multi_nli_model, train_loader, optimizer, loss_fn, y_train, batch_size)
    validation_matched_predictions = multi_nli_model_eval(device, multi_nli_model, validation_matched_loader, \
                                                          optimizer, loss_fn, y_validation_matched, \
                                                          'validation_matched')
    validation_mismatched_predictions = multi_nli_model_eval(device, multi_nli_model, validation_mismatched_loader, \
                                                             optimizer, loss_fn, y_validation_mismatched, \
                                                             'validation_mismatched')
    print ('Validation Matched Data ROC AUC Score :-')
    print (roc_auc_score(torch.tensor(validation_matched_predictions).cpu().data.numpy(), \
                         torch.tensor(y_validation_matched).cpu().data.numpy()))
    print ('Validation Matched Data Confusion Matrix :-')
    print (confusion_matrix(torch.tensor(validation_matched_predictions).cpu().data.numpy(), \
                            torch.tensor(y_validation_matched).cpu().data.numpy()))
    print ('Validation Mismatched Data ROC AUC Score :-')
    print (roc_auc_score(torch.tensor(validation_mismatched_predictions).cpu().data.numpy(), \
                         torch.tensor(y_validation_mismatched).cpu().data.numpy()))
    print ('Validation Mismatched Data ROC AUC Score :-')
    print (confusion_matrix(torch.tensor(validation_mismatched_predictions).cpu().data.numpy(), \
                            torch.tensor(y_validation_mismatched).cpu().data.numpy()))


'''
The controller function of the script
'''
def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    glove = GloVe(name='6B', dim=100)
    sst_utils(device, lemmatizer, stop_words, glove)
    multi_nli_utils(device, lemmatizer, stop_words, glove)


if __name__ == '__main__':
    main()
                
