# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""================================================================
Example usage:
python /Users/sswpro/Documents/CalWaterPolo/object_detection/tfrecord/create_wp_tfr_eval.py --game_name=20180127_CHNvCAL

-----------------------------------------------------------------
In the project we need clips in "clips" folder, json files in "composite_jsons" folder 
and wp_label_map.pbtxt with create_wp_tf_record.py.

  EX)
  ## clips
  /Users/sswpro/Documents/object_detection/tfrecord/clips/20180127_CHNvCAL

  ## jsons
  /Users/sswpro/Documents/object_detection/tfrecord/composite_jsons/20180127_CHNvCAL

  ## label_map
  /Users/sswpro/Documents/object_detection/tfrecord/wp_label_map.pbtxt

================================================================
#game_name is the name of game you want to evaluate
EX) 20180127_CHNvCAL

================================================================
================================================================
#output tfrecord dir
/Users/sswpro/Documents/object_detection/tfrecord/output/20180127_CHNvCAL
"""

import tensorflow as tf
import io
import random
import json
import hashlib

import PIL.Image
from os import getcwd, listdir, makedirs
from os.path import isfile, join, exists

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('proj_dir', '', 'Path to the project that has clips,jsons and label_map')
flags.DEFINE_string('game_name', '', 'Name of the wp game')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def sample_dataset(clip_path):
  image_files = [file for file in listdir(clip_path) if isfile(join(clip_path, file))]
  random.seed(49)
  random.shuffle(image_files)
  num_images = len(image_files)
  num_eval = int(0.3 * num_images)
  eval_clips = image_files[:num_eval]
  return eval_clips, num_eval

def file_name_padding(filename):
  num_padding = '0000'
  file_name = filename.split('.')[0]
  put_zeros = 4 - len(file_name)
  filename = (num_padding[:put_zeros] + file_name + '.jpg').encode('utf8')
  return filename

def dict_to_tf_example(data, detections, label_map_dict, clip_directory):
  #Populate the following variables
  height = data['height'] # Image height
  width = data['width'] # Image width
  filename = file_name_padding(data['image_file']) # Filename of the image. Empty if image is not from file
  img_path = join(clip_directory, filename)

  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_image_data = fid.read() # Encoded image bytes
  encoded_img_io = io.BytesIO(encoded_image_data)
  image = PIL.Image.open(encoded_img_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_image_data).hexdigest()

  # image_format = data['image_file'].split('.')[1] # b'jpeg' or b'png'
  image_format = 'jpg'
  
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  # print("height, width: ", height, width)
  for detection in detections:
 
    ymin = detection['bounding_box'][0]/height
    xmin = detection['bounding_box'][1]/width
    ymax = detection['bounding_box'][2]/height
    xmax = detection['bounding_box'][3]/width

    xmins.append(xmin) 
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)
    class_num = detection['class']
    classes.append(class_num)
    for key in label_map_dict.keys():
      if label_map_dict[key] == class_num:
        classes_text.append(key.encode('utf8'))
        # break
    # classes_text.append(class_name.encode('utf8'))
    # classes.append(label_map_dict[class_num])

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def create_tf_record(output_filename, label_map_dict, 
      json_path, clip_path, clips):
  """Creates a TFRecord file from clips.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    json_path: Path to json files are stored.
    clip_path: Path to image files are stored.
    clips: clips to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  with open(json_path) as f:
    data = json.load(f)

  for clip in clips:
    clip_name = clip.split('.')[0]
    uclip = unicode(clip_name, "utf-8")
    detections = data['detections'].get(uclip, None)
    if not detections:
      continue
    tf_example = dict_to_tf_example(data, detections, label_map_dict, clip_path)
    writer.write(tf_example.SerializeToString())

  writer.close()

def main(_):
  proj_dir = getcwd()
  game_name = FLAGS.game_name

  image_dir = join(proj_dir, 'clips', game_name)
  json_dir = join(proj_dir, 'composite_jsons', game_name)
  label_map_dir = join(proj_dir, 'wp_label_map.pbtxt')
  label_map_dict = label_map_util.get_label_map_dict(label_map_dir)
  output_dir = join(proj_dir, 'output')

  if not exists(output_dir):
    makedirs(output_dir)

  total_train_num = 0
  total_eval_num = 0

  json_files = [file for file in listdir(json_dir) if isfile(join(json_dir, file))]
  for json_file in json_files:
    clip_num = json_file.split('.')[0]
    clip_path = join(image_dir ,clip_num)
    json_path = join(json_dir, json_file)

    train_output_name = join(output_dir, 'wp_train.record')
    eval_output_name = join(output_dir, 'wp_eval.record')

    eval_clips, eval_num = sample_dataset(clip_path)
    total_train_num += train_num
    total_eval_num += eval_num
    create_tf_record(eval_output_name, label_map_dict, 
      json_path, clip_path, eval_clips)

  print("total_eval_num: ", total_eval_num)
if __name__ == '__main__':
  tf.app.run()