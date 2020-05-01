import torch

class opts:
    MAX_LENGTH = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze = False
    scale_grad_by_freq = True
    teacher_forcing_ratio = 0.3

    input_file = '/home/artin/Projects/VizWiz/vizwiz-caption/annotations/train_split.json'
    val_file = '/home/artin/Projects/VizWiz/vizwiz-caption/annotations/val_split.json'

    input_bu_folder = '/home/artin/Projects/VizWiz/final_vizwiz_data/train_36'
    val_bu_folder = '/home/artin/Projects/VizWiz/final_vizwiz_data/val_36'
    no_boxes = 36
    vis_size = 2048

    input_size = 300
    hidden_size = 300  # 256
    run_name = 'v14'