import json
import os
import cPickle as pickle
train_image_path = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
valid_image_path = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_validation_images_20170902/'
data_path = '/media/dl/expand/data/'

train_caption_file = data_path + 'caption_train_annotations_cut.json'
valid_caption_file = data_path + 'caption_valid_annotations_cut.json'


def generate_ftoid(image_path, prefix):
    filelist = os.popen('ls ' + image_path).readlines()
    filenames = [image[:-1] for image in filelist]
    print filenames[0]
    print filenames[-1]
    filename_to_id = {}
    for i, filename in enumerate(filenames):
        filename_to_id[filename] = i
    pickle.dump(filename_to_id, file('./%s_filename_to_id.pkl' % (prefix), 'w'))


def split_captions(train_num):
    # split dataset into training and validation subset, train_num is the size of training subset
    with open(caption_file) as f:
        caption_data = json.load(f)
    assert len(caption_data) >= train_num
    train_split = []
    valid_split = []
    for i in xrange(train_num):
        train_split.append(caption_data[i])
    for i in xrange(train_num, len(caption_data)):
        valid_split.append(caption_data[i])
    print 'size of train_split:', len(train_split)
    print 'size of valid_split:', len(valid_split)
    return train_split, valid_split


def generate_captions(caption_file, filename_to_id):
    caption_list = []
    with open(caption_file) as f:
        caption_data = json.load(f)
    for item in caption_data:
        image_id = item['image_id']
        captions = item['caption']
        for caption in captions:
            caption_dict = {}
            caption_dict['image_id'] = filename_to_id[image_id]
            caption_dict['file_name'] = image_id
            caption_dict['caption'] = caption
            caption_list.append(caption_dict)

    return caption_list


if __name__ == '__main__':
    for subdir in ['train', 'validation']:
        print subdir, '---------'
        if not os.path.exists(data_path + subdir):
            os.makedirs(data_path + subdir)
        image_path = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_%s_images_20170902/' % (subdir)
        caption_file = data_path + 'caption_%s_annotations_cut.json' % (subdir)
        generate_ftoid(image_path, subdir)
        filename_to_id = pickle.load(file('./%s_filename_to_id.pkl' % (subdir), 'r'))
        captions = generate_captions(caption_file, filename_to_id)
        json.dump(captions, file(data_path + '%s/%s_captions.json' % (subdir, subdir), 'w'))
