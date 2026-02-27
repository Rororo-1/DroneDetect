import glob
import tensorflow as tf
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

images = glob.glob("dataset/images/*.png")
labels = glob.glob("dataset/labels/*.txt")

images.sort()
labels.sort()

def load(images):
    def decode(image):
        image_dat = tf.io.read_file(image)
        image = tf.image.decode_png(image_dat, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = preprocess_input(image)
        return image
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=(64))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
images_ds = load(images)

base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
base.trainable = False

features = base.predict(images_ds)


cluster_model = KMeans(n_clusters=13, random_state=1)
skal_features = StandardScaler().fit_transform(features)
clusters = cluster_model.fit_predict(skal_features)

print(silhouette_score(skal_features, clusters))

for label, cluster in zip(labels, clusters):
    x = []
    with open(label, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            parts[0] = str(cluster)
            x.append(' '.join(parts))
    with open(label, 'w') as file:
        file.write(' '.join(x))
