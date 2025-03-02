import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
torch.manual_seed(12)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    print("vec")
    print(vec.shape)
    print("vec: ", vec)
    max_score, max_indices = torch.max(vec, dim=-1, keepdim=True)
    print("max score: ", max_score)
    max_score_broadcast = max_score.expand_as(vec)
    print("max score broadcast: ", max_score_broadcast)


    return (max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))).unsqueeze(-1)
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim) # normal word embed
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first = True) # normal bilstm with batch false
        # the input shape will be seq_len, batchsize, embed dim
        # I added batch first = True. thus it will be batch_size, seq_len, embed_dim

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # automatically handles batch size due to projection

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(    # nn.parameters means that these parameters will be updated through gradient descent
            torch.randn(self.tagset_size, self.tagset_size)) #independent of batch

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000 # independent of batch
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000 # independent of batch

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2)) # intialize for random weights. Independent of batching

    def _forward_alg(self, feats):
        print("inside forward algo")
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000.)
        print("init alphas ")
        print(init_alphas)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        print("init alphas after 0ing start")
        print(init_alphas)
        print("shape feats ")
        print(feats.shape)
        print("feats ")
        print(feats)


        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        batch_size, seq_len, tag_dim = feats.shape
        for sequence_token in range(seq_len):
            feat = feats[:, sequence_token, :] # processing across all batches
            alphas_t = None  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # next tag will be the shape batch size
                print("shape feat next_tag, then actual feat next tag")
                print(feat[:, next_tag].shape)
                print(feat[:, next_tag])  #
                emit_score = feat[:, next_tag].unsqueeze(1).expand(batch_size, self.tagset_size) # broadcasts the scalar emit score to be the size of the tag_set
                # we do this because the emission score is the same regardless of the previous tag
                print("shape emit score")
                print(emit_score.shape)
                print(emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i

                trans_score = self.transitions[next_tag].view(1, -1).expand(batch_size, self.tagset_size)
                # expand broadcasts it accross the batches
                print("trans score")
                print(trans_score)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                print("🥭🥭forward var shape, trans var shape, emit score shape:")
                print(forward_var.shape, trans_score.shape, emit_score.shape)
                next_tag_var = forward_var + trans_score + emit_score
                print("next tag var shape")
                print(next_tag_var.shape)
                print("next tag var")
                print(next_tag_var)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.

                log_sum_next_var = log_sum_exp(next_tag_var)
                print("log sum next var")
                print(log_sum_next_var.shape,log_sum_next_var)
                if alphas_t is None:
                    alphas_t = log_sum_next_var
                else:
                    alphas_t = torch.cat((alphas_t, log_sum_next_var), dim=1)  # this is the total sum of getting to the next point
                print("alpha_ts.shape")
                print(alphas_t.shape)
                print("printing alpha ts")
                print(alphas_t)


            forward_var = alphas_t # its already in the proper tensor so we dont need to worry about this
            print("forward var shape")
            print(forward_var.shape)
            print("printing forward var")
            print(forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        print("FINAL ALPHA: ")
        print(alpha.view(1, -1).squeeze()) # batch size, 1
        return alpha.view(1, -1).squeeze()  # gets the probability for the sentence given ALL the tag pats

    def _get_lstm_features(self, batch):

        print("batch.shape: ", batch.shape)
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(batch)
        print("embeds.shape: ", embeds.shape)
        # self.word_embeds(sentence) = batch_size, seq_length,embedding_dim

        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # batch_size, seq len, hidden dim
        print("lstm_out.shape: ", lstm_out.shape)
        lstm_feats = self.hidden2tag(lstm_out) # batch_size, tagset size
        print("☘️☘️☘️ printing lstm_feats shape ☘️☘️☘️")
        print(lstm_feats.shape)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        print("we are inside score sentence")
        batch_size, seq_len, tag_dim = feats.shape

        score = torch.zeros(batch_size)
        tags = torch.cat([torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long), tags], dim=1)
        print("tags")
        print(tags)
        print(tags.shape)
        # adds start token to all seq in batch
        for i in range(seq_len):
            feat = feats[:, i, :] # minic the loop in the non batched version
            print("new feat shape")
            print(feat.shape)
            print(feat)
            print("self.transitions[tags[:, i + 1], tags[:, i]]")
            print(self.transitions[tags[:, i + 1], tags[:, i]])
            print(self.transitions[tags[:, i + 1], tags[:, i]].shape)

            score = score + self.transitions[tags[:, i + 1], tags[:, i]] + feat.gather(1, tags[:, i + 1].unsqueeze(1)).squeeze(1)
            print("intermediate score: ", score) # calculates the core over just the correct tag sequence

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        print("final score: ", score)

        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.batch_size, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        per_seq_score = forward_score - gold_score
        print("we are investigating loss")
        print(per_seq_score.shape)
        print(sum(per_seq_score))
        return sum(per_seq_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 128




def get_data_from_iob(path):

    with open(path, 'r') as file:
        iob_data = file.read()


    sentences = iob_data.strip().split('\n\n') # sentences in data split by 2 lines

    data = []

    for sentence in sentences:

        tokens_tags = sentence.split('\n') # gets all the tokens in the sentence
        tokens = []
        tags = []

        for token_tag in tokens_tags:

            token, tag = token_tag.split('\t')

            tokens.append(token)
            tags.append(tag)


        data.append([tokens, tags])


    df = pd.DataFrame(data, columns=['tokens', 'tags'])


    #df.to_csv('iob_to_csv.csv', index=False)

    return df

train_df = get_data_from_iob("A2-data/train")
dev_df = get_data_from_iob("A2-data/dev.answers")
test = get_data_from_iob("A2-data/test.answers")
word_to_ix = {}



class IOBDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.vocab = {"PAD": 0, "UNK": 1}
        self.tag_vocab = {"PAD": 0, "UNK": 1, "<START>": 2, "<STOP>": 3}
        self.x = []
        self.y = []
        self.inverse_vocab = {}
        self.inverse_tag_vocab = {}
    def build_vocab(self, df):


        for row in df.itertuples():


            for token in row.tokens:
                if token not in self.vocab:

                    self.vocab[token] = len(self.vocab)
            for tag in row.tags:
                if tag not in self.tag_vocab:
                    self.tag_vocab[tag] = len(self.tag_vocab)
        #print(self.vocab)
        #print(self.tag_vocab)
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}
        self.inverse_tag_vocab = {value: key for key, value in self.tag_vocab.items()}




    def load_vocab(self, vocab, tag_vocab):
        self.vocab = vocab
        self.inverse_vocab = {value: key for (key,value) in vocab.items()}
        self.tag_vocab = tag_vocab
        self.inverse_tag_vocab = {value: key for (key, value) in tag_vocab.items()}


    def encode_data(self, df):
        counter = 0
        for row in df.itertuples():

            if counter == 8:
                break
            counter += 1
            row_x = []
            row_y = []
            for token in row.tokens:
                row_x.append(self.vocab.get(token, 1))
            for tag in row.tags:
                row_y.append(self.tag_vocab.get(tag, 1))
            self.x.append(torch.tensor(row_x))
            self.y.append(torch.tensor(row_y))
    def decode_data(self, idx):
        tokens = ""
        tags = []

        for token in self.x[idx]:
            tokens += self.inverse_vocab.get(token.item(), 1) + " "
        for tag in self.y[idx]:

            tags.append(self.inverse_tag_vocab.get(tag.item(), 1))
        return tokens, tags
    def check_work(self, idx):
        if idx > len(self.x):
            idx = len(self.x) -1
        tokens, tags = self.decode_data(idx)
        print(tokens)
        print(tags)


    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


train_dataset = IOBDataset()
train_dataset.build_vocab(train_df)

train_dataset.encode_data(train_df)
train_dataset.check_work(500)

dev_dataset = IOBDataset()
dev_dataset.build_vocab(dev_df)

dev_dataset.encode_data(dev_df)
dev_dataset.check_work(500)

test_dataset = IOBDataset()
test_dataset.build_vocab(dev_df)
test_dataset.encode_data(dev_df)
test_dataset.check_work(500)
batch_size = 8

import torch.nn.functional as F
def collate_fn(batch):

    inputs, labels = zip(*batch) # both are a list of torch tensors


    max_len = max(seq.shape[0] for seq in inputs)

    padded_inputs = torch.stack([F.pad(seq, (0, max_len - seq.shape[0]), value=0) for seq in inputs])

    labels = torch.stack([F.pad(seq, (0, max_len - seq.shape[0]), value=0) for seq in labels])


    print(padded_inputs.shape)
    print(labels.shape)

    return padded_inputs, labels
print("number of x and y")
print(len(train_dataset.x))
print(len(train_dataset.y))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = BiLSTM_CRF(len(train_dataset.vocab), train_dataset.tag_vocab, EMBEDDING_DIM, HIDDEN_DIM, 8)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training



# Make sure prepare_sequence from earlier in the LSTM section is loaded
print(len(train_loader))
for epoch in range(
        1):
    for batch in train_loader:
            sentence,tags = batch
            print("shape of sentence and tag in the loop")
            print(sentence.shape)
            print(tags.shape)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()


            loss = model.neg_log_likelihood(sentence, tags)
            print(loss)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!'''