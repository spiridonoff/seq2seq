import torch
import torch.nn as nn
import torch.nn.functional as F
from opts import opts
import random

opt = opts
device = opt.device
freeze = opt.freeze
scale_grad_by_freq = opt.scale_grad_by_freq
teacher_forcing_ratio = opt.teacher_forcing_ratio
SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, w2v):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(w2v, freeze=freeze, scale_grad_by_freq=scale_grad_by_freq)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vis_size, no_boxes, w2v, dropout_p=0.1,
                 max_length=opt.MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vis_size = vis_size
        self.no_boxes = no_boxes
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding.from_pretrained(w2v, freeze=freeze, scale_grad_by_freq=scale_grad_by_freq)
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.vis_attn = nn.Linear(self.hidden_size + self.no_boxes * self.vis_size, self.no_boxes)  # + self.input_size
        self.vis_avgpool = nn.AvgPool1d(2, stride=2)
        self.vis_attn_combine = nn.Linear(self.hidden_size + int(self.vis_size / 2), self.hidden_size)
        self.attn_combine = nn.Linear(self.input_size + int(self.hidden_size / 2), self.input_size)  # + self.vis_size
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, vis_features):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        vis_attn_weights = F.softmax(
            self.vis_attn(torch.cat((hidden[0], vis_features.view(1, -1)), dim=1)), dim=1)
        vis_attn_applied = torch.bmm(vis_attn_weights.unsqueeze(0), vis_features.unsqueeze(0))
        vis_attn_pooled = self.vis_avgpool(vis_attn_applied)

        hidden = torch.cat((hidden[0], vis_attn_pooled[0]), 1)
        hidden = self.vis_attn_combine(hidden).unsqueeze(0)
        hidden = F.relu(hidden)

        output = torch.cat((embedded[0], attn_applied[0]), 1)  # , vis_attn_applied[0]
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights  # , vis_attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sent):
    idx = [lang.word2index[word] for word in sent.split()]
    return idx


def tensorFromSentence(lang, sent1, sent2 = None):
    indexes1 = indexesFromSentence(lang, sent1)
    indexes2 = []
    if sent2:
        indexes2 = indexesFromSentence(lang, sent2)
        indexes2.append(EOS_token)
    else:
        indexes1.append(EOS_token)
    return torch.tensor(indexes1, dtype=torch.long, device=device).view(-1, 1), torch.tensor(indexes2, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, lang):
    input1_tensor, input2_tensor = tensorFromSentence(lang, pair[0], pair[1])
    target_tensor, _ = tensorFromSentence(lang, pair[2])
    return input1_tensor, input2_tensor, target_tensor


def train(input1_tensor, input2_tensor, vis_tensor, target_tensor, encoder1, encoder2, decoder, encoder1_optimizer,
          encoder2_optimizer, decoder_optimizer, criterion, max_length=opt.MAX_LENGTH):
    encoder1_hidden = encoder1.initHidden()
    encoder2_hidden = encoder2.initHidden()

    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input1_length = input1_tensor.size(0)
    input2_length = input2_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder1_outputs = torch.zeros(int(max_length / 2), encoder1.hidden_size, device=device)
    encoder2_outputs = torch.zeros(int(max_length / 2), encoder2.hidden_size, device=device)

    loss = 0

    for ei in range(input1_length):
        encoder1_output, encoder1_hidden = encoder1(
            input1_tensor[ei], encoder1_hidden)
        encoder1_outputs[ei] = encoder1_output[0, 0]

    for ei in range(input2_length):
        encoder2_output, encoder2_hidden = encoder2(
            input2_tensor[ei], encoder2_hidden)
        encoder2_outputs[ei] = encoder2_output[0, 0]

    encoder_outputs = torch.cat((encoder1_outputs, encoder2_outputs), 0)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    #     decoder_hidden = encoder_hidden ###
    decoder_hidden = torch.cat((encoder1_hidden, encoder2_hidden), 2)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, vis_tensor)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, vis_tensor)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder1_optimizer.step()
    encoder2_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length