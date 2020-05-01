import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import os
import shutil

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = 'vizwiz_train_36.tsv'
output_dir = 'train_36'

os.makedirs(output_dir + '_box')

if __name__ == '__main__':
# Verify we can read a tsv
    no_items = 0
    in_data = {}
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            no_items += 1
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), dtype=np.float32).reshape((item['num_boxes'],-1))
            
            image_id = item['image_id']
            if infile == 'vizwiz_val_36.tsv':
                image_id += 23431
            
            in_data[image_id] = item
            np.savez_compressed(os.path.join(output_dir, str(image_id)), feat=item['features'])
#             np.save(os.path.join(output_dir + '_box', str(image_id)), item['boxes'])
                    
    print('number of items: %d'%no_items)
                    