import tensorflow as tf

data = ["This is toy example number " + str(n) for n in range(1,101)]

def create_example(text):
    # Create a dictionary mapping the feature name to the tf.train.Feature
    feature = {
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode()]))
    }
    
    # Create a Features message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for text in data:
            example = create_example(text)
            writer.write(example)

# Specify the TFRecord filename
filename = 'toy_dataset.tfrecord'

# Write the dataset to a TFRecord file
write_tfrecords(data, filename)
