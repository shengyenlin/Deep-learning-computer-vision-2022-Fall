import json

train_text_path = './hw3_data/p2_data/train.json'
val_text_path = './hw3_data/p2_data/val.json'
train_text_out_path = './hw3_data/p2_data/train_organized.json'
val_text_out_path = './hw3_data/p2_data/val_organized.json'

def organize_data(in_path, out_path):
    f = open(in_path)
    data = json.load(f)
    annot = data['annotations']
    image = data['images']
    imageid2filename = dict()
    for data in image:
        imageid2filename[data['id']] = data['file_name']

    out_data = dict()
    for data in annot:
        id = data['id']
        data_dict = dict()
        file_name = imageid2filename[data['image_id']]
        data_dict['caption'] = data['caption']
        data_dict['image_id'] = data['image_id']
        data_dict['file_name'] = file_name
        out_data[id] = data_dict
    
    with open(out_path, 'w') as f:
        json.dump(out_data, f)


def main():
    organize_data(train_text_path, train_text_out_path)
    organize_data(val_text_path, val_text_out_path)

if __name__ == '__main__':
    main()