import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
torch.manual_seed(12)
from torch.utils.data import Dataset

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
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
        return forward_score - gold_score

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
        self.tag_vocab = {"PAD": 0, "UNK": 1}
        self.x = []
        self.y = []
        self.inverse_vocab = {}
        self.inverse_tag_vocab = {}
    def build_vocab(self, df):

        for row in df.itertuples():
            print(row)
            for token in row.tokens:
                if token not in self.vocab:
                    print(token)
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
        for row in df.itertuples():
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

        return len(self.vocab)

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


def collate_fn(batch):

    inputs, labels = zip(*batch)

    max_len = max(len(seq) for seq in inputs)

    padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]

    padded_inputs = torch.tensor(padded_inputs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_inputs, labels



model = BiLSTM_CRF(len(word_to_ix), len(train_dataset), EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training



# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10):
    for sentence, tags in batch:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!'''