import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
torch.manual_seed(12)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    #print("vec")
    #print(vec.shape)

    max_score, max_indices = torch.max(vec, dim=-1, keepdim=True)

    max_score_broadcast = max_score.expand_as(vec)



    return (max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))).unsqueeze(-1)
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size, char_vocab_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.char_embed_model = Character_CNN(char_vocab_dim)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # normal word embed
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first = True, dropout= .4) # normal bilstm with batch false
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




    def forward(self, char_embeds):
        """
        char_embeds: (batch, word_len, char_embedding_dim)
        """
        char_embeds = char_embeds.permute(0, 2, 1)  # (batch, char_embedding_dim, word_len) для Conv1d
        conv_out = self.conv1d(char_embeds)  # Применяем свертку
        conv_out = self.relu(conv_out)  # Активация
        pooled_out = self.pool(conv_out)  # Применяем макспулинг

        # Агрегация: берем max по последней оси, чтобы получить фиксированный вектор на слово
        char_features = torch.max(pooled_out, dim=-1)[0]  # (batch, cnn_output_dim)

        return char_features


    def _forward_alg(self, feats, get_all_scores = False):
        #print("inside forward algo")
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000.)
        #print("init alphas ")
        #print(init_alphas)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        #print("init alphas after 0ing start")
        #print(init_alphas)
        #print("shape feats ")
        #print(feats.shape)
        #print("feats ")
        #print(feats)


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
                #print("shape feat next_tag, then actual feat next tag")
                #print(feat[:, next_tag].shape)
                #print(feat[:, next_tag])  #
                emit_score = feat[:, next_tag].unsqueeze(1).expand(batch_size, self.tagset_size) # broadcasts the scalar emit score to be the size of the tag_set
                # we do this because the emission score is the same regardless of the previous tag
                #print("shape emit score")
                #print(emit_score.shape)
                #print(emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i

                trans_score = self.transitions[next_tag].view(1, -1).expand(batch_size, self.tagset_size)
                # expand broadcasts it accross the batches
                #print("trans score")
                #print(trans_score)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                #print("🥭🥭forward var shape, trans var shape, emit score shape:")
                #print(forward_var.shape, trans_score.shape, emit_score.shape)
                next_tag_var = forward_var + trans_score + emit_score
                #print("next tag var shape")
                #print(next_tag_var.shape)
                #print("next tag var")
                #print(next_tag_var)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.

                log_sum_next_var = log_sum_exp(next_tag_var)
                #print("log sum next var")
                #print(log_sum_next_var.shape,log_sum_next_var)
                if alphas_t is None:
                    alphas_t = log_sum_next_var
                else:
                    alphas_t = torch.cat((alphas_t, log_sum_next_var), dim=1)  # this is the total sum of getting to the next point
                #print("alpha_ts.shape")
                #print(alphas_t.shape)
                #print("printing alpha ts")
                #print(alphas_t)


            forward_var = alphas_t # its already in the proper tensor so we dont need to worry about this
            #print("forward var shape")
            #print(forward_var.shape)
            #print("printing forward var")
            #print(forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        #print("FINAL ALPHA: ")
        #print(alpha.view(1, -1).squeeze()) # batch size, 1
        if get_all_scores:
            return terminal_var
        else:
            return alpha.view(1, -1).squeeze()  # gets the probability for the sentence given ALL the tag pats

    def _get_lstm_features(self, batch, chars):

        char_embeds = self.char_embed_model(chars)


        #print("batch.shape: ", batch.shape)
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(batch)
        embeds = embeds + char_embeds
        #print("embeds.shape: ", embeds.shape)
        # self.word_embeds(sentence) = batch_size, seq_length,embedding_dim

        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # batch_size, seq len, hidden dim
        #print("lstm_out.shape: ", lstm_out.shape)
        lstm_feats = self.hidden2tag(lstm_out) # batch_size, tagset size
        #print("☘️☘️☘️ printing lstm_feats shape ☘️☘️☘️")
        #print(lstm_feats.shape)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        #print("we are inside score sentence")
        batch_size, seq_len, tag_dim = feats.shape

        score = torch.zeros(batch_size)
        tags = torch.cat([torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long), tags], dim=1)
        #print("tags")
        #print(tags)
        #print(tags.shape)
        # adds start token to all seq in batch
        for i in range(seq_len):
            feat = feats[:, i, :] # minic the loop in the non batched version
            #print("new feat shape")
            #print(feat.shape)
            #print(feat)
            #print("self.transitions[tags[:, i + 1], tags[:, i]]")
            #print(self.transitions[tags[:, i + 1], tags[:, i]])
            #print(self.transitions[tags[:, i + 1], tags[:, i]].shape)

            score = score + self.transitions[tags[:, i + 1], tags[:, i]] + feat.gather(1, tags[:, i + 1].unsqueeze(1)).squeeze(1)
            #print("intermediate score: ", score) # calculates the core over just the correct tag sequence

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        #print("final score: ", score)

        return score

    def _get_hamming_score(self, feats, tags):
        best_path = self._viterbi_decode(feats)
        mask = tags != 0

        total = mask.sum().item()
        incorrect = (best_path != tags) & mask

        hamming_loss = incorrect.sum().item() / total if total > 0 else 0.0
        return hamming_loss

    def _viterbi_decode(self, feats):
        #print("\n\n😏😏======== BEGINNING VITERBI ==========😏😏")
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.batch_size, self.tagset_size), -10000.)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        batch_size, seq_len, tagset_dim = feats.shape
        for seq_token in range(seq_len):
            #print("\n\nSTARTING A NEW TOKEN ~~~~~ ")
            feat = feats[:, seq_token, :] # we iterate over sequence length
            #print("feat.shape: ", feat.shape)
            bptrs_t = None  # holds the backpointers for this step
            # we will make this start as none. At the first interation we will create it and then we will keep appending to it.
            # it will be of shape batch_size, tagset_size
            viterbivars_t = None  # holds the viterbi variables for this step
            # we will make this start as none. At the first interation we will create it and then we will keep appending to it.
            # it will be of shape batch_size, tagset_size

            for next_tag in range(self.tagset_size): # this is the tag that we are transitioning too
                #print("\n\nSTARTING A NEW TAG ")
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)

                # forward var is dim batch_size, tagset size. Transitions[next_tag] will be of dim (, tagset_size), so we need to broadcast


                next_tag_var = forward_var + self.transitions[next_tag].expand(batch_size, tagset_dim)
                #print("next tag var shape, ", next_tag_var.shape)
                best_tag_id = torch.argmax(next_tag_var, dim=-1) # we are only saving the best path (through the best past tag) to get to the current tag
                # we will need argmaxes for each batch because it will be batchsize, tagset dim
                # best tag id should be of dim batchsize, 1
                #print("++best tag id shape, ", best_tag_id.shape)
                if bptrs_t is None:
                    bptrs_t = best_tag_id.unsqueeze(1)
                    #print("shape back pointers after init: ", bptrs_t.shape)
                else:
                    bptrs_t = torch.cat((bptrs_t, best_tag_id.unsqueeze(1)), dim=-1)# # saves this information. We do this because we want to follow the trail of backpointers for the one that ends in stop
                    # we are appending it along the sequence len dimebsion
                    #print("shape back pointers after cat: ", bptrs_t.shape)

                if viterbivars_t is None:
                    viterbivars_t = next_tag_var.gather(1, best_tag_id.unsqueeze(1))
                    #print("shape viterbivars pointers after init: ", viterbivars_t.shape)

                else:
                    #print("the shape of the next best tags: ", next_tag_var.gather(1, best_tag_id.unsqueeze(1)).shape)
                    viterbivars_t = torch.cat((viterbivars_t, next_tag_var.gather(1, best_tag_id.unsqueeze(1))), dim=-1)# gets the max by indexing at the max index. Then tranposes to be of shape(1)
                    #print("shape viterbivars pointers after cat: ", viterbivars_t.shape)

                # so this is the actual max probability before adding the emission prob

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t) # addend all backpointers for that token
            # this will have (batchsize, tagset size tensors for every token in sequence )

        #print("\n\n 🍋WE ARE STARTING TO REVERSE🍋 ")
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].expand(batch_size, tagset_dim)
        #print("terminal_var.shape: ", terminal_var.shape)
        best_tag_id = torch.argmax(terminal_var, dim=-1)
        #print("best_tag_id.shape: ", best_tag_id.shape)
        path_score = terminal_var[:, best_tag_id]

        # Follow the back pointers to decode the best path.

        best_path = best_tag_id.view(batch_size, 1)# batchsize, 1
        for bptrs_t in reversed(backpointers): # iterating through the list of batchsize by tagset size tensors
            # bptrs_t = shape(batch_size, tagset size)
            #print("\n\nstarting new back pointer")

            #print("bptrs_t.shape: ", bptrs_t.shape)
            best_tag_id = torch.gather(bptrs_t, 1, best_tag_id.unsqueeze(1)).squeeze()
            #print("best tag id shape in the loop: ", best_tag_id.shape)
            viewed_best_tag_id = best_tag_id.view(batch_size, 1)
            #print("__best_tag_id.shape:, ", viewed_best_tag_id.shape)
            #print("best_path shape: ", best_path.shape)
            best_path = torch.cat((viewed_best_tag_id, best_path), dim=-1) # concats it to the path
            #print("we completed a cycle")
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path[:, 0] # we appended in reverse order so we check the 0th tag

        decoded_path = []
        ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}

        for seq in best_path:
            decoded_tags = [ix_to_tag[idx.item()] for idx in seq]
            decoded_path.append(decoded_tags)

        # Print the decoded paths
        #print("Decoded best path: ", decoded_path[0])



        assert (start == self.tag_to_ix[START_TAG]).all(), "some sequences don't start with START_TAG!"  # Sanity check

        return path_score, best_path[:, 1:]

    def neg_log_likelihood(self, sentence, chars, tags):
        feats = self._get_lstm_features(sentence, chars)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        per_seq_score = forward_score - gold_score
        #print("we are investigating loss")
        #print(per_seq_score.shape)
        #print(sum(per_seq_score))
        return sum(per_seq_score)

    def _ramp_loss(self, sentence, chars, tags):
        """
        Computes the ramp loss using terminal scores from the forward algorithm,
        ignoring padding positions where true tags are 0.

        Args:
            feats (Tensor): Emission scores [batch_size, seq_len, num_tags]
            tags (Tensor): True tag sequences [batch_size, seq_len]

        Returns:
            Tensor: Ramp loss value [batch_size]
        """
        feats = self._get_lstm_features(sentence, chars)
        # Create a mask where true tags are not padding (0)
        mask = tags != 0  # shape: [batch_size, seq_len]

        # Compute terminal scores using the forward algorithm
        terminal_scores = self._forward_alg(feats, get_all_scores=True)  # shape: [batch_size, num_tags]

        # Compute the best sequence using Viterbi decoding
        best_path = self._viterbi_decode(feats)  # shape: [batch_size, seq_len]
        best_path_score = self._score_sentence(feats, best_path)  # shape: [batch_size]

        # Compute the Hamming loss, ignoring padding positions
        hamming_loss = torch.tensor(
            [self._get_hamming_score(feats[i], tags[i]) for i in range(feats.shape[0])],
            device=feats.device, dtype=feats.dtype
        )  # shape: [batch_size]

        # Compute the worst violating sequence score
        worst_violation_scores = terminal_scores + hamming_loss.unsqueeze(1)  # Add cost
        worst_path_score = worst_violation_scores.max(dim=1)[0]  # Take max violation

        # Compute ramp loss
        loss = -best_path_score + worst_path_score  # Ramp loss formula

        # Apply mask to ignore padding loss
        loss = loss * mask.any(dim=1)  # Only keep nonzero losses per sequence

        return loss.mean()  # Average over batch


    def add_batch_predictions_to_export_file(self, sentence, chars, export_file_path):

            feats = self._get_lstm_features(sentence, chars)

            path_score, best_path = self._viterbi_decode(feats)  # shape: [batch_size, seq_len]

            print(best_path.shape)


            flattened_predictions = best_path.reshape(-1, 1)  # shape: [batch_size * seq_len, 1]


            df_predictions = pd.DataFrame(flattened_predictions.cpu().numpy(), columns=["Predictions"])
            print(len(df_predictions))

            df_predictions.to_csv(export_file_path, mode='a', header=False, index=False)

            print(f"Batch predictions added to {export_file_path}")

    def add_true_y_to_export_file(self, tags, export_file_path):


            print(tags.shape)
            flattened_trues= tags.reshape(-1, 1)  # shape: [batch_size * seq_len, 1]

            df_predictions = pd.DataFrame(flattened_trues.cpu().numpy(), columns=["Predictions"])
            print(len(df_predictions))


            df_predictions.to_csv(export_file_path, mode='a', header=False, index=False)

            print(f"Batch predictions added to {export_file_path}")

    def _svm_loss(self, sentence, chars, tags):

        feats = self._get_lstm_features(sentence, chars)  # [batch_size, seq_len, num_tags]

        mask = tags != 0  # [batch_size, seq_len] (mask padded positions)


        gold_score = self._score_sentence(feats, tags)  # [batch_size]

        # gets the terminal scores from forward algorithm (ending prob of each path)
        terminal_scores = self._forward_alg(feats, get_all_scores=True)  # [batch_size, num_tags]

        # get the hamming cost (ignoring padding)
        hamming_loss = self._get_hamming_score(feats, tags)

        # get the max probability with the hamming loss
        max_margin_violation = (terminal_scores + hamming_loss).max(dim=1)[0]  # [batch_size]

        # SVM loss
        loss = max_margin_violation - gold_score

        # mask where the true tags  are 0)
        loss = loss * mask.any(dim=1).float()

        return loss.mean()  # averaging over batch

    def forward(self, sentence, chars):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, chars)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 32
HIDDEN_DIM = 32


class Character_CNN(nn.Module):
    def __init__(self, char_vocab_size):
        super(Character_CNN, self).__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, 64, padding_idx=0)


        self.conv1d = nn.Conv1d(in_channels=64,
                                out_channels=32,
                                kernel_size=3,
                                padding=1)

        self.relu = nn.ReLU()

    def forward(self, char_inputs):
        # char_inputs - (batch_size, word_seq_len, char_seq_len)
        char_inputs = char_inputs.long()
        #print("char_inputs.shape")

        #print(char_inputs.shape)
        batch_size, word_seq_len, char_seq_len = char_inputs.shape


        char_embeds = self.char_embedding(char_inputs)  # (batch_size, word_seq_len, char_seq_len, char_embedding_dim)

        # changing the dimensions for convolution
        char_embeds = char_embeds.permute(0, 3, 1, 2)  # (batch_size, char_embedding_dim, word_seq_len, char_seq_len)
        #print("char_embeds.shape after permute:", char_embeds.shape)


        conv_out = self.conv1d(char_embeds.reshape(batch_size * word_seq_len, 64, char_seq_len))  # (batch_size * word_seq_len, 32, new_char_seq_len)
        conv_out = self.relu(conv_out)
        #print("conv_out.shape after conv1d:", conv_out.shape)

        # put it back to the old dim
        conv_out = conv_out.view(batch_size, word_seq_len, 32, -1)  # (batch_size, word_seq_len, 32, new_char_seq_len)

        # max pool on the characters in the word
        char_features = torch.max(conv_out, dim=-1)[0]  # (batch_size, word_seq_len, 32)
        #print("char_features.shape:", char_features.shape)

        return char_features


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
test_df = get_data_from_iob("A2-data/test.answers")
word_to_ix = {}



class IOBDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.vocab = {"PAD": 0, "UNK": 1}
        self.tag_vocab = {"PAD": 0, "UNK": 1, "<START>": 2, "<STOP>": 3}
        self.x = []
        self.x_char = [] # for our character embeddings
        self.y = []

        self.inverse_vocab = {}
        self.inverse_tag_vocab = {}
        self.char_vocab = {"PAD": 0, "UNK": 1, "_": 2}
        self.test_sentence = None # stores a batch of 8 test sentences for debugging
        self.test_tags = None
    def build_vocab(self, df):


        for row in df.itertuples():


            for token in row.tokens:
                if token not in self.vocab:

                    self.vocab[token] = len(self.vocab)
                for letter in token:
                    if letter not in self.char_vocab:
                        self.char_vocab[letter] = len(self.char_vocab)
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
            if counter == 1000:
                break
            else:
                counter +=1



            row_x = []
            row_x_char = []
            row_y = []


            max_char_len = 0
            for token in row.tokens:
                max_char_len = max(max_char_len, len(token))

            for token in row.tokens:
                token_char = []
                row_x.append(self.vocab.get(token, 1))


                for char in token:
                    token_char.append(self.char_vocab.get(char, 1))


                token_char.extend([self.char_vocab.get('<PAD>', 0)] * (
                            max_char_len - len(token_char)))

                row_x_char.append(torch.tensor(token_char))

            for tag in row.tags:
                row_y.append(self.tag_vocab.get(tag, 1))


            self.x.append(torch.tensor(row_x))
            self.x_char.append(torch.stack(
                row_x_char))
            self.y.append(torch.tensor(row_y))

        test_sentence = ["CD28", "activity", "is", "useful"]

        test_tags = ["B-protein", "O", "O", "O"]
        sen = []
        sen_chars = []
        tag = []
        for token in test_sentence:
            sen.append(self.vocab.get(token, 1))

            for char in token:
                sen_chars.append(self.char_vocab.get(char, 1))

        for token in test_tags:
            tag.append(self.tag_vocab.get(token, 1))
        sen_tensor = torch.tensor(sen, dtype=torch.long)
        char_tensor = torch.tensor(sen_chars, dtype=torch.long)
        tag_tensor = torch.tensor(tag, dtype=torch.long)
        self.test_sentence = sen_tensor.unsqueeze(0).repeat(8, 1)  # (8, len(sen))
        self.test_chars = char_tensor.unsqueeze(0).repeat(8,1)
        self.test_tags = tag_tensor.unsqueeze(0).repeat(8, 1)




    # decodes based on index
    def decode_data(self, idx):
        tokens = ""
        tags = []

        for token in self.x[idx]:
            tokens += self.inverse_vocab.get(token.item(), 1) + " "
        for tag in self.y[idx]:

            tags.append(self.inverse_tag_vocab.get(tag.item(), 1))
        return tokens, tags

    # decodes sentence
    def decode_sentence(self, sentence, tags_seq):
        sentence = sentence.squeeze() # size seq_len
        print(sentence.shape)
        tags_seq = tags_seq.squeeze() # size seq_len
        print(tags_seq.shape)
        tokens = ""
        tags = []

        for token in sentence:
            tokens += self.inverse_vocab.get(token.item(), 1) + " "
        for tag in tags_seq:

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

        return self.x[idx], self.x_char[idx], self.y[idx]


train_dataset = IOBDataset()
train_dataset.build_vocab(train_df)

train_dataset.encode_data(train_df)
train_dataset.check_work(500)

dev_dataset = IOBDataset()
dev_dataset.build_vocab(train_df)

dev_dataset.encode_data(dev_df)
dev_dataset.check_work(500)

test_dataset = IOBDataset()
test_dataset.build_vocab(train_df)
test_dataset.encode_data(test_df)
test_dataset.check_work(500)
batch_size = 8

import torch.nn.functional as F


def collate_fn(batch):
    inputs, char_inputs, labels = zip(*batch)
    # char input is a tuple with 8 elements each representing a seq in the batch
    #print(char_inputs[0].shape) # the tensors inside the tuple have shape [seq_len, character_len)
    # character len has been padded per sentence, but not per batch
    #print(char_inputs[1].shape)
    #print(char_inputs[2].shape)
    #print(char_inputs[3].shape)
    #print(char_inputs[4].shape)
    #print(char_inputs[5].shape)
    #print(char_inputs[6].shape)
    #print(char_inputs[7].shape)



    # the max characters in a word
    max_len = max(seq.shape[0] for seq in inputs)
    max_char_len_per_word = max(
        max(word.shape[0] for word in seq) if len(seq) > 0 else 0
        for seq in char_inputs
    )
    #print("max_char_len_per_word:", max_char_len_per_word)

    # normal pad for words
    padded_inputs = torch.stack([F.pad(seq, (0, max_len - seq.shape[0]), value=0) for seq in inputs])

    # normal pad for tags
    labels = torch.stack([F.pad(seq, (0, max_len - seq.shape[0]), value=0) for seq in labels])

    # word level padding for character inputs

    #print("\n\nstarting to pad the characters to be the batch max")
    batched_chars = []
    for seq in char_inputs:
        #print("\n\n ☘️☘️ starting new seq")
        padded_words = []
        for word in seq:

            if word.shape[0] < max_char_len_per_word:
                padding_size = max_char_len_per_word - word.shape[0]  # the difference between itself and the max char len
                padding = torch.zeros(padding_size)  # creates the vector of 0s to concat
                padded_word = torch.cat([word, padding], dim=0)
            else:
                padded_word = word

            padded_words.append(padded_word)
        new_seq = torch.stack(padded_words)
        #print("🥭🥭old_seq_shape: ", seq.shape)
        #print("🥭🥭new_seq.shape: ", new_seq.shape)


        # nice! now are characters are padded. Now we just need to pad along batch
        full_pad_word = torch.zeros(max_char_len_per_word) # for when we need to add words full of just pad chars
        if new_seq.shape[0] < max_len: # if the number of words in the seq is less than the total number of words
            amount_of_zero_rows_to_add = max_len - new_seq.shape[0]
            zero_rows = torch.stack(
                [full_pad_word for _ in range(amount_of_zero_rows_to_add)])  # make the 0 tensor

            padded_seq = torch.cat([new_seq, zero_rows], dim=0)  # add the 0 rows along the sequence dim
            #print("old shape: ", new_seq.shape)
            #print("padded_seq shape:", padded_seq.shape)
        else:
            padded_seq = new_seq
        batched_chars.append(padded_seq)




    padded_char_inputs = torch.stack(batched_chars)

    #print("padded_char_inputs.shape: ", padded_char_inputs.shape)
    return padded_inputs, padded_char_inputs, labels

print("number of x and y")
print(len(train_dataset.x))
print(len(train_dataset.y))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = BiLSTM_CRF(len(train_dataset.vocab), train_dataset.tag_vocab, EMBEDDING_DIM, HIDDEN_DIM, 8, char_vocab_dim=len(train_dataset.char_vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


import os

base_dir = os.getcwd()

folder = os.path.join(base_dir, "svm_loss")

os.makedirs(folder, exist_ok=True)

val_file = os.path.join(folder, "val_predictions.csv")
test_file = os.path.join(folder, "test_predictions.csv")
val_file_true = os.path.join(folder, "val_true.csv")
test_file_true = os.path.join(folder, "test_true.csv")

'''with torch.no_grad():
    print("we are doing precheck")
    precheck_sent = train_dataset.test_sentence
    precheck_chars = train_dataset.test_chars
    print(precheck_sent)
    print(model(precheck_sent, precheck_chars))'''

# Make sure prepare_sequence from earlier in the LSTM section is loaded
print(len(train_loader))
best_val_loss = float("inf")
best_model = None
for epoch in range(4):
    print("STARTING EPOCH: ", epoch)
    epoch_loss = []
    for batch in train_loader:
            sentence, chars, tags = batch
            if sentence.shape[0] != 8:
                continue
            else:
                #print("shape of sentence and tag in the loop")
                #print(sentence.shape)
                #print(tags.shape)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()


                loss = model._svm_loss(sentence, chars, tags)

                epoch_loss.append(loss.item())
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
    print("printing the last train prediction of batch")
    sentence = sentence[0, :]
    tags = tags[0, :]
    chars = chars[0, :]
    _sentence, _tags = train_dataset.decode_sentence(sentence, tags)
    print("true sentence:")
    print(_sentence)
    print("true tags:")
    print(_tags)
    print("predicted tags:")
    model(sentence.repeat(8, 1), chars.repeat(8,1, 1)) # reapeating it so that it doesnt piss off our lstm'''
    epoch_loss = np.mean(epoch_loss)
    print("THE EPOCH LOSS WAS: ", epoch_loss)

    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            sentence, chars, tags = batch
            if sentence.shape[0] != 8:
                continue
            else:
                #print("shape of sentence and tag in the loop")
                #print(sentence.shape)
                #print(tags.shape)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()


                loss = model._svm_loss(sentence, chars, tags)

                val_loss.append(loss.item())
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
    print("printing the last val prediction")
    sentence = sentence[0, :]
    tags = tags[0, :]
    chars = chars[0, :]
    _sentence, _tags = train_dataset.decode_sentence(sentence, tags)
    print("true sentence:")
    print(_sentence)
    print("true tags:")
    print(_tags)
    print("predicted tags:")
    model(sentence.repeat(8, 1), chars.repeat(8, 1, 1))  # reapeating it so that it doesnt piss off our lstm'''

    val_loss = np.mean(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()


    print("THE VAL LOSS WAS: ", val_loss)




print("training done!")

model.load_state_dict(best_model)
if 5 > 2: # do not want to have to undo all the indents
    test_loss = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sentence, chars, tags = batch
            if sentence.shape[0] != 8:
                continue
            else:
                #print("shape of sentence and tag in the loop")
                #print(sentence.shape)
                #print(tags.shape)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()


                loss = model._svm_loss(sentence, chars, tags)
                model.add_batch_predictions_to_export_file(sentence, chars, test_file)
                model.add_true_y_to_export_file(tags, test_file_true)
                test_loss.append(loss.item())
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()

    test_loss = np.mean(test_loss)
    print("THE TEST LOSS WAS: ", test_loss)

    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            sentence, chars, tags = batch
            if sentence.shape[0] != 8:
                continue
            else:
                # print("shape of sentence and tag in the loop")
                # print(sentence.shape)
                # print(tags.shape)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                loss = model._svm_loss(sentence, chars, tags)
                model.add_batch_predictions_to_export_file(sentence, chars, val_file)
                model.add_true_y_to_export_file(tags, val_file_true)
                val_loss.append(loss.item())
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()

    val_loss = np.mean(val_loss)