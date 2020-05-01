from __future__ import unicode_literals, print_function, division
from io import open
import pickle
# import torch
# import torch.nn as nn
from torch import optim
from utils import *
from models import *

from opts import opts
opt = opts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainEpochs(encoder1, encoder2, decoder, n_epochs, print_every=5000, plot_every=1000, learning_rate=0.01, max_wait=3):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    iter = 0
    min_loss = 10.0
    waited = 0

    encoder1_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.SGD(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    lmbda = lambda epoch: 0.95
    scheduler_encoder1 = optim.lr_scheduler.MultiplicativeLR(encoder1_optimizer, lr_lambda=lmbda, last_epoch=-1)
    scheduler_encoder2 = optim.lr_scheduler.MultiplicativeLR(encoder2_optimizer, lr_lambda=lmbda, last_epoch=-1)
    scheduler_decoder = optim.lr_scheduler.MultiplicativeLR(decoder_optimizer, lr_lambda=lmbda, last_epoch=-1)
    
#     swa_start = 300 #(n_epochs-3)*n_pairs
#     swa_freq = 20 #20000
#     swa_lr = learning_rate/2
    
#     encoder1_optimizer = SWA(encoder1_optimizer_base, swa_start = swa_start, swa_freq = swa_freq, swa_lr = swa_lr)
#     encoder2_optimizer = SWA(encoder2_optimizer_base, swa_start = swa_start, swa_freq = swa_freq, swa_lr = swa_lr)
#     decoder_optimizer = SWA(decoder_optimizer_base, swa_start = swa_start, swa_freq = swa_freq, swa_lr = swa_lr)

    #     training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    n_pairs = len(pairs)
    n_iters = n_epochs * n_pairs
    print('NO. iterations in each epoch: %d' % n_pairs)

    for epoch in range(n_epochs):
        epoch_loss_total = 0.0
        for pair in pairs:  #  ########<<<<<<<<<<< remember to revert this
            iter += 1
            training_pair = tensorsFromPair(pair, lang)  # training_pairs[iter - 1]
            input1_tensor = training_pair[0]
            input2_tensor = training_pair[1]
            target_tensor = training_pair[2]
            image_id = pair[3]
            vis_tensor = get_bu_tensor(opt.input_bu_folder, image_id)

            loss = train(input1_tensor, input2_tensor, vis_tensor, target_tensor, encoder1, encoder2,
                         decoder, encoder1_optimizer, encoder2_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            epoch_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                showPlot(plot_losses)

        epoch_loss_avg = epoch_loss_total / len(pairs)
        print('epoch: %d, loss: %.4f' % (epoch, epoch_loss_avg))

        if epoch == 0:
            min_loss = epoch_loss_avg
            torch.save(encoder1.state_dict(), './models/encoder1_checkpoint.pth')
            torch.save(encoder2.state_dict(), './models/encoder2_checkpoint.pth')
            torch.save(attn_decoder1.state_dict(), './models/decoder_checkpoint.pth')
            print('new checkpoint saved. epoch:%d, loss:%.4f' % (epoch, epoch_loss_avg))

        elif epoch_loss_avg < min_loss:
            min_loss = min(epoch_loss_avg, min_loss)
            waited = 0
            # save checkpoint
            torch.save(encoder1.state_dict(), './models/encoder1_checkpoint.pth')
            torch.save(encoder2.state_dict(), './models/encoder2_checkpoint.pth')
            torch.save(attn_decoder1.state_dict(), './models/decoder_checkpoint.pth')
            print('new checkpoint saved. epoch:%d, loss:%.4f' % (epoch, epoch_loss_avg))

        elif waited + 1 < max_wait:
            waited += 1
        else:
            print('Early-stopping is terminating the training at epoch: %d. Loading the last checkpoint...' % epoch)
            #             encoder1.load_state_dict(torch.load('./encoder1_checkpoint.pth'))
            #             encoder2.load_state_dict(torch.load('./encoder2_checkpoint.pth'))
            #             attn_decoder1.load_state_dict(torch.load('./decoder_checkpoint.pth'))
            #             print('last checkpoint loaded.')
            break
        
        scheduler_encoder1.step()
        scheduler_encoder2.step()
        scheduler_decoder.step()

    pickle.dump(plot_losses, open('./plots/plot_losses_%s' % opt.run_name, 'wb'))
    showPlot(plot_losses)
    print('Training finished.')

    encoder1.load_state_dict(torch.load('./models/encoder1_checkpoint.pth'))
    encoder2.load_state_dict(torch.load('./models/encoder2_checkpoint.pth'))
    attn_decoder1.load_state_dict(torch.load('./models/decoder_checkpoint.pth'))
    print('last checkpoint loaded.')


Data = pickle.load(open('language.pkl', 'rb'))
lang = Data['lang']
pairs = Data['pairs']
w2v = Data['w2v']
del Data

encoder1 = EncoderRNN(lang.n_words, opt.hidden_size, w2v).to(device)
encoder2 = EncoderRNN(lang.n_words, opt.hidden_size, w2v).to(device)
attn_decoder1 = AttnDecoderRNN(opt.input_size, 2*opt.hidden_size, lang.n_words, opt.vis_size, opt.no_boxes,
                               w2v, dropout_p=0.1).to(device)

trainEpochs(encoder1, encoder2, attn_decoder1, 12, print_every=5000, plot_every=1000, learning_rate=0.01, max_wait=4)

torch.save(encoder1.state_dict(), './models/encoder1_%s.pth'% opt.run_name)
torch.save(encoder2.state_dict(), './models/encoder2_%s.pth' % opt.run_name)
torch.save(attn_decoder1.state_dict(), './models/decoder_%s.pth' % opt.run_name)