import tensorflow.compat.v2 as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from preprocessDefinition import preprocess

# file below will be replaced by a testing set of identical format
data, info = tfds.load(name='oxford_flowers102', split=['train', 'validation'],as_supervised=True,with_info=True)
train_ds = data[0]
valid_ds = data[1]
dataset=train_ds.map(preprocess)
dataset_valid=valid_ds.map(preprocess)

base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
avg=keras.layers.GlobalAveragePooling2D()(base_model.output)
output=keras.layers.Dense(102,activation="softmax")(avg)
model=keras.models.Model(inputs=base_model.input,outputs=output)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,
              metrics=['accuracy',top1err,top5err,top10err])

model.fit(dataset_tr,validation_data=dataset_valid_fit,epochs=25,
          callbacks=[checkpoint_cb,earlyStop_cb])

#returns the loss and metrics evaluated on the dataset
model.evaluate(dataset_ts)
model.save("flowersModel.h5")
