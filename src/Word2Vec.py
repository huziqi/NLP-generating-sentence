import word2vec

word2vec.word2phrase('/home/guohf/AI_tutorial/ch8/data/text8','/home/guohf/AI_tutorial/ch8/data/text8-phrases',verbose=True)
word2vec.word2vec('/home/guohf/AI_tutorial/ch8/data/text8-phrases','/home/guohf/AI_tutorial/ch8/data/text8.bin', size=100, verbose=True)
word2vec.word2clusters('/home/guohf/AI_tutorial/ch8/data/text8','/home/guohf/AI_tutorial/ch8/data/text8-clusters.txt',100,verbose=True)


model=word2vec.load('/home/guohf/AI_tutorial/ch8/data/text8.bin')
# take look at the vocabulary as a numpy array
print(model.vocab)

# take a look at the whole matrix
print(model.vectors.shape)
print(model.vectors)

# retreive the vector of individual words
print(model['dog'].shape)
print(model['dog'][:10])

# calculate the distance between two or more(all combinations) words
print(model.distance("dog","cat","fish"))

# do simple queries to retreive words similar to "dog" based on cosine similarity
indexes, metrics=model.similar("dog")
print(indexes,metrics)

# to get those words retreived
print(model.vocab[indexes])

# There is a helper function to create a combined response as a numpy record array
model.generate_response(indexes, metrics)

# to make that numpy array a pure python response
model.generate_response(indexes, metrics).tolist()

# Its possible to do more complex queries like analogies such as: king - man + woman = queen This method returns
# the same as cosine the indexes of the words in the vocab and the metric
indexes, metrics = model.analogy(pos=['king', 'woman'], neg=['man'])
print(indexes, metrics)


clusters = word2vec.load_clusters('/home/guohf/AI_tutorial/ch8/data/text8-clusters.txt')

# get the cluster number for individual words
print(clusters.vocab)

# We can see get all the words grouped on an specific cluster
print(clusters.get_words_on_cluster(90).shape)
print(clusters.get_words_on_cluster(90)[:10])

# add the clusters to the word2vec model and generate a response that includes the clusters
model.clusters = clusters
indexes, metrics = model.analogy(pos=["paris", "germany"], neg=["france"])
model.generate_response(indexes, metrics).tolist()


