import os
import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
from opts import opts
opt = opts


def get_bu_tensor(folder, image_id, no_boxes=36, vis_size=2048):
    file_name = str(image_id) + '.npz'
    vis_array = np.load(os.path.join(folder, file_name))['feat']

    if vis_array.shape[1] != vis_size:
        print('features for image {} do not have proper size. {}'.format(image_id, vis_array.shape))

    if vis_array.shape != (no_boxes, vis_size):
        vis_array_full = np.zeros((no_boxes, vis_size), dtype=np.float32)
        for i in range(vis_array.shape[0]):
            vis_array_full[i] = vis_array[i]
    else:
        vis_array_full = vis_array
    return torch.from_numpy(vis_array_full).to(opt.device)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
    file_name = 'losses_%s.png' % opt.run_name
    plt.plot(points)
    plt.savefig('./plots/%s' % file_name)
    plt.close(fig)