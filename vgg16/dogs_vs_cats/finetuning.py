# -*- coding: utf-8 -*-
import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from smallcnn import save_history
from keras.initializers import he_normal, glorot_normal
import subprocess
import keras
import argparse
from tensorflow.python.lib.io import file_io
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

img_width, img_height = 60, 60
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50
batch_size=64
result_dir = '/tmp'
MODEL = 'dor_or_cat.hdf5'

def go_main(job_dir, gs_download):
    if job_dir.startswith('gs://') or gs_download == 'True':
        cmd = 'gsutil -m cp -r gs://kceproject-1113-ml/cat_dog/data.tar.gz /tmp'
        subprocess.check_call(cmd.split())
        cmd = 'tar -zxvf /tmp/data.tar.gz -C /tmp'
        subprocess.check_call(cmd.split())
        train_data_dir = '/tmp/data/train'
        validation_data_dir = '/tmp/data/validation'
    else:
        train_data_dir = '/home/jiman/cat_dog/data/train'
        validation_data_dir = '/home/jiman/cat_dog/data/validation'
    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(256, activation='relu', kernel_initializer=he_normal()))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_normal()))

    # 学習済みのFC層の重みをロード
    # TODO: ランダムな重みでどうなるか試す
    #top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    # vgg16_modelはkeras.engine.training.Model
    # top_modelはSequentialとなっている
    # ModelはSequentialでないためadd()がない
    # そのためFunctional APIで二つのモデルを結合する
    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    model.summary()

    # layerを表示
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:18]:
        layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    model.summary()

    # TODO: ここでAdamを使うとうまくいかない
    # Fine-tuningのときは学習率を小さくしたSGDの方がよい？
    model.compile(loss='binary_crossentropy',
                  #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  optimizer=optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        #rescale=1.0 / 255,
        #shear_range=0.2,
        #zoom_range=0.2,
        preprocessing_function=preprocess_input,
        horizontal_flip=True)

    #test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [tblog]

    # Fine-tuning
    train_steps = int(nb_train_samples/batch_size)+1
    val_steps = int(nb_validation_samples/batch_size)+1
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=callbacks)

    #model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
    #save_history(history, os.path.join(result_dir, 'history_finetuning.txt'))

    if job_dir.startswith("gs://"):
        model.save(MODEL)
        copy_file_to_gcs(job_dir, MODEL)
    else:
        model.save(os.path.join(job_dir, MODEL))


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') \
                as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and '
                        'export model')
    parser.add_argument('--gs-download',
                        type=str,
                        help='GCS download or not',
                        )
    parse_args, unknown = parser.parse_known_args()
    go_main(**parse_args.__dict__)
