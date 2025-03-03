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
    vec = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    print("vec: ", vec)
    max_score, max_indices = torch.max(vec, dim=-1, keepdim=True)
    print("max score: ", max_score)
    max_score_broadcast = max_score.expand_as(vec)
    print("max score broadcast: ", max_score_broadcast)

    print(max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1)))
    return max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1)) # we are going out of logspace temporarily because log(a+b) =/ log(a) + log(b)
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
        print("inside forward algo")
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        print(init_alphas)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        print(init_alphas)
        print(feats)

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                print("shape feat next_tag, then actual feat next tag")
                print(feat[next_tag].shape)
                print(feat[next_tag]) #
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size) # broadcasts the scalar emit score to be the size of the tag_set
                # we do this because the emission score is the same regardless of the previous tag
                print("shape emit score")
                print(emit_score.shape)
                print(emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                print("☘️ self.transitions")
                print(self.transitions)
                print("☘️ self.transitions[next_tag]")
                print(self.transitions[next_tag])
                trans_score = self.transitions[next_tag].view(1, -1)
                print(trans_score)
                print("☘️ shape trans score: ", trans_score.shape)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp

                next_tag_var = forward_var + trans_score + emit_score
                print("next tag var shape")
                print(next_tag_var.shape)
                print("next tag var")
                print(next_tag_var)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                print("shape before view:")
                print(log_sum_exp(next_tag_var))
                print("shape after view")
                print(log_sum_exp(next_tag_var).view(1))
                alphas_t.append(log_sum_exp(next_tag_var).view(1)) # this is the total sum of getting to the next point
                print("alpha_ts.shape")
                print(len(alphas_t))
                print("printing alpha ts")
                print(alphas_t)
            print("printing alpha ts")
            print(alphas_t)
            print("printing torch.cat(alphas_t)")
            print(torch.cat(alphas_t))

            forward_var = torch.cat(alphas_t).view(1, -1)
            print("forward var shape")
            print(forward_var.shape)
            print("printing forward var")
            print(forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        return alpha # gets the probability for the sentence given ALL the tag pats

    def _get_lstm_features(self, sentence):
        print("inside get lstm features")
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        print("shape lstm out")
        print(lstm_out.shape)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        print("shape lstm out")
        print(lstm_out.shape)
        lstm_feats = self.hidden2tag(lstm_out)
        print("shape lstm feats")
        print(lstm_feats.shape)
        print("exiting get lstm features")
        return lstm_feats

    def _score_sentence(self, feats, tags):
        print("inside score sentenc")
        # Gives the score of a provided tag sequence
        print("feat shape")
        print(feats.shape)
        print(feats)

        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        print("printing tags in score function")
        print(tags)
        for i, feat in enumerate(feats):
            print("shape of this feat")
            print(feat.shape)
            print("self.transitions[tags[i + 1]")
            print(self.transitions[tags[i + 1], tags[i]])
            print("feat[tags[i + 1]]")
            print(feat[tags[i + 1]])

            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            print("printing score inside score function")
            print(score)
            print("score shape!")
            print(score.shape)
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        print("final score shape!: ", score.shape)
        return score # gets the probability for the sentence given only the best tag path

    def _viterbi_decode(self, feats):
        backpointers = []
        print("inside viterbi!")

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        print("init vars shape")
        print(init_vvars.shape)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        print("feats shape")
        print(feats.shape)
        for feat in feats: # this iterates over sequence length
            print("the shape of this feat")
            print(feat.shape)
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)

                # forward var is the probability of the past token
                next_tag_var = forward_var + self.transitions[next_tag] # gets the transitions from all possible past vars
                print("self.transitions next var: ", self.transitions[next_tag])

                best_tag_id = argmax(next_tag_var) # gets the tag that lead to the best path
                bptrs_t.append(best_tag_id) # saves this information. We do this because we want to follow the trail of backpointers for the one that ends in stop

                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) # gets the max by indexing at the max index. Then tranposes to be of shape(1)
                # so this is the actual max probability before adding the emission prob
                # is a one dim tensor

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            print("viterbi vars!")
            print(viterbivars_t)
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
        print("inside neg log like")
        print(sentence)
        feats = self._get_lstm_features(sentence)
        gold_score = self._score_sentence(feats, tags)
        print("printing gold score in neg log like")

        forward_score = self._forward_alg(feats)
        print("printing forward score in neg log lik")
        print(forward_score)

        print(gold_score)
        return forward_score - gold_score # calculates how different the score is when considering all tags to the score when only considering the true
        # idea is that loss of 0 means the probabilities of the true tag sequence is 1 and probabilities of other tags are 0

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)



# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        1):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        print("about to run prepare_sequence")
        sentence_in = prepare_sequence(sentence, word_to_ix)
        print("preparing targets")
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        print("running neg log like")
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!
