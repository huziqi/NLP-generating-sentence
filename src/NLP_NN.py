import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

class Dataloader():
    def __init__(self):
        #path= tf.keras.utils.get_file('nietzsche.txt',
        #                             origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        path = '/home/guohf/old_man_and_sea.txt'
        with open(path, encoding='utf-8') as f:
            self.raw_text= f.read().lower()
        self.chars= sorted(list(set(self.raw_text)))
        self.char_indices= dict((c,i) for i, c in enumerate(self.chars))
        self.indices_char= dict((i,c) for i, c in enumerate(self.chars))
        self.text= [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, batch_size):
        first_word= []
        next_word= []
        for i in range(batch_size):
            index= np.random.randint(0, len(self.text)-3-batch_size)
            first_word.append(self.text[index:index+3])
            next_word.append(self.text[index+3])
        return np.array(first_word), np.array(next_word)

class NN(tf.keras.Model):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars= num_chars
        self.embeded= tf.keras.layers.Embedding(input_dim=self.num_chars, output_dim=100)
        self.dense1= tf.keras.layers.Dense(units=500)
        self.dense2= tf.keras.layers.Dense(units=self.num_chars)
        self.flatten_= tf.keras.layers.Flatten()

    def call(self, inputs):
        inputs= tf.one_hot(inputs, depth=self.num_chars)
        print("shape of one-hot:", tf.shape(inputs))
        x = self.embeded(inputs)
        print("shape after embeding:", tf.shape(x))
        x = self.flatten_(x)
        print("shape after flatten:", tf.shape(x))
        x=self.dense1(x)
        print("shape after dense1:", tf.shape(x))
        output= self.dense2(x)
        print("shape of outputs:", tf.shape(output))
        return output

    def predict(self, inputs, temperature=1.):
        batch_size = tf.shape(inputs)
        logits=self(inputs)
        prob= tf.nn.softmax(logits/ temperature).numpy()
        print(prob)
        return np.array([np.random.choice(self.num_chars, p=prob[0, :])])


num_batches=5
batch_size=50
learning_rate=0.001
print("start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

data_loader=Dataloader()
model=NN(len(data_loader.chars))
checkpoint= tf.train.Checkpoint(NN_100000= model)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred= model(X)
        #print("shape of y:", tf.shape(y))
        #print("shape of y_logit_pred:", tf.shape(y_logit_pred))
        loss= tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads= tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

checkpoint.save('/home/guohf/AI_tutorial/ch8/model/model_nn_100000.ckpt')

X_, _=data_loader.get_batch(1)
for diversity in [0.2,0.5,1.0,1.2,1.5,2]:
    X=X_
    print("diversity %f:" % diversity)
    for t in range(100):
        y_pred= model.predict(X,diversity)
        print(data_loader.indices_char[y_pred[0]], end='',flush=True)
        #print("shape of X:", X)
        X=np.concatenate([X[:,1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("/n")