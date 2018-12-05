import os
import shutil
import argparse
import csv
import numpy as np
import tensorflow as tf


# get file path form command parameters

parser = argparse.ArgumentParser()
# Required arguments
parser.add_argument("--output_path", type=str)
parser.add_argument("--csv_file", type=str)
args, unknown_args = parser.parse_known_args()

CSV_FILE = args.csv_file
OUTPUT_PATH = args.output_path


# add middle layers (unusable)

#def inference(x_ph):
#    hidden1 = tf.layers.dense(x_ph, 32, activation=tf.nn.relu)
#    hidden2 = tf.layers.dense(hidden1, 32, activation=tf.nn.relu)
#    logits = tf.layers.dense(hidden2, 3)
#    return logits


# load and split the csv data

def load_data():
    
    with tf.gfile.Open(CSV_FILE) as f:
        reader = csv.reader(f)
        mat = np.array([row for row in reader][1:], dtype=np.float)
    
    ind_train = np.random.choice(600, 450, replace=False)
    ind_test = np.array([i for i in range(600) if i not in ind_train])
    
    x_train = mat[ind_train, :-1]
    x_test = mat[ind_test, :-1]
    
    y_all = np.zeros([len(mat), 6])
    for i, j in enumerate(mat[:, -1]):
        y_all[i][j-1] = 1.
    
    y_train = y_all[ind_train]
    y_test = y_all[ind_test]

    return x_train, y_train, x_test, y_test


# main process

x_train, y_train, x_test, y_test = load_data()

with tf.Graph().as_default() as g:

    tf.set_random_seed(0)

    # �]�g�i�w�K�j�f�[�^��x�A����i�\���j���ʂ�y�Ɋi�[����
    x = tf.placeholder(tf.float32, [None, 5])
    
    # �d�݂�W�A�o�C�A�X��b�Ƃ��ĕϐ��錾
    W = tf.Variable(tf.zeros([5, 6]))
    b = tf.Variable(tf.zeros([6]))

    # �g���[�j���O�f�[�^�isoftmax�֐��ŏ����j
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # logits = inference(x_ph)
    # y = tf.nn.softmax(logits)
    
    # �������x���f�[�^
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    # �����֐��i�N���X�G���g���s�[�j
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_ph, logits=logits, label_smoothing=1e-5)
    
    # ���z�~���@��p���ăN���X�G���g���s�[���ŏ�������
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
    
    # �\���l�Ɛ���l���r����bool�l�ɂ���
    # argmax(y, 1)�͗\���l�̊e�s�ōő�ƂȂ�C���f�b�N�X���ЂƂԂ�
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # boole�l��0��������1�ɕϊ����ĕ��ϒl���Ƃ�A����𐳉𗦂Ƃ���
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # �ϐ��̏�����
    init = tf.global_variables_initializer()
    
    # �Z�b�V�����̊J�n
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(1000):
        
            # �g���[�j���O�f�[�^���烉���_����100���o
            ind = np.random.choice(len(y_train), 100)
            # �m���I���z�~���@�ɂ��N���X�G���g���s�[���ŏ�������悤�ȏd�݂��X�V����
            sess.run(train_step, feed_dict={x: x_train[ind], y_: y_train[ind]})
            
            if i % 100 == 0:
                train_loss = sess.run(cross_entropy, feed_dict={x: x_train, y_: y_train})
                train_accuracy, y_pred = sess.run([accuracy, y], feed_dict={x: x_train, y_: y_train})
                test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
                tf.logging.info(
                    "Iteration: {0} Loss: {1} Train Accuracy: {2} Test Accuracy{3}".format(
                        i, train_loss, train_accuracy, test_accuracy
                    )
                )
        
        # if not os.path.isdir("checkpoints"):
        #     os.mkdir("checkpoints")
        saver.save(sess, "{}/checkpoints/emotionanalysis".format(OUTPUT_PATH))
        
        # Save model for deployment on ML Engine
        input_key = tf.placeholder(tf.int64, [None, ], name="key")
        output_key = tf.identity(input_key)
        input_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(input_key),
            "x": tf.saved_model.utils.build_tensor_info(x)
        }
        output_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(output_key),
            "y": tf.saved_model.utils.build_tensor_info(y)
        }
        predict_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            input_signatures,
            output_signatures,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder = tf.saved_model.builder.SavedModelBuilder("{}/model".format(OUTPUT_PATH))
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
        )
        builder.save()
