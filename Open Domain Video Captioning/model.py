import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

class V2D():
    def __init__(self, dim_image, vocabulary, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image      # dimension of the image feature extracted from the cnn
        self.vocabulary = vocabulary    # number of words
        self.dim_hidden = dim_hidden    # memory state of lstm, also dimension of the words in vocabulary
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step

        #with tf.device("/gpu:2"):
            # for every word we need to have a word embedding of 1000 dimensions (dim-hidden)
        #if isTrain:
        self.Wemb = tf.Variable(tf.random_uniform([vocabulary, dim_hidden], -0.1, 0.1), name='Wemb')
        #else:	
	#    self.Wemb = tf.get_variable('Wemb', shape = [vocabulary, dim_hidden])
        # create 2 LSTM cells with 1000 hidden units
        # state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. If False,
        # they are concatenated along the column axis.
        self.lstm1 = tf.nn.rnn_cell.LSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.LSTMCell(dim_hidden, state_is_tuple=False)

        # encode the 4096 dimensional dim_image feature vector
        # for LSTM 1
        #if isTrain:
        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
	#else:
        #    self.encode_image_W = tf.get_variable(shape = [dim_image, dim_hidden], name="encode_image_W")
	#    self.encode_image_b = tf.get_variable(shape = [dim_hidden], name='encode_image_b')
        #for LSTM 2 we define the weights and biases
        #if isTrain:
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, vocabulary], -0.1, 0.1), name='embed_word_W')
	#else:
        #    self.embed_word_W = tf.get_variable(shape = [dim_hidden, vocabulary], name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([vocabulary]), name='embed_word_b')

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
    	video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

    	# dimension reshaped to (80, 4096)
    	video_flat = tf.reshape(video, [-1, self.dim_image])
         
    	 # do the matrix multiplication operation and addition of biases
    	 # encode_image_W has dimension = (4096,1000)
    	 # encode_image_b has dimension = (1000)
    	 # video_flat has shape = (80, 4096)	
    	 # obtained dimension = (80, 1000)
    	image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
	    
    	 # reshape back to (1,80,1000)
    	image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])
	 
    	state1 = tf.zeros([1, self.lstm1.state_size])
    	state2 = tf.zeros([1, self.lstm2.state_size])
    	padding = tf.zeros([1, self.dim_hidden])
	 
    	 # stores the max probabilty word index in the vocabulary for all the timesteps
    	 # dimenstion - (20, 1)
    	generated_words = []
	 
    	 # stores the logit words that is the probability of all the words in the vocab for all the timesteps
    	 # dimension - (20, vocabulary)
    	probs = []
	 
    	 # stores the word embedding of the words for all the timesteps 
    	 # dimension - (20, 1000)
    	embeds = []
	 
    	for i in range(0,self.n_video_lstm_step):
    	    if i > 0:
	        tf.get_variable_scope().reuse_variables()
	
            with tf.variable_scope("LSTM1"):
    	        output1, state1 = self.lstm1(image_emb[:, i, :], state1)
	 
    	    with tf.variable_scope("LSTM2"):
    	        output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
	 
    	for i in range(0, self.n_caption_lstm_step):
       	    if i == 0:
         	    #with tf.device('/gpu:2'):
		# find the word embedding for [1] 
		# 1 because index 1 of the vocabulary is <bos>
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            tf.get_variable_scope().reuse_variables()
 
    	    with tf.variable_scope("LSTM1"):
    	        output1, state1 = self.lstm1(padding, state1)
	 
    	    with tf.variable_scope("LSTM2"):
    	        output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)
	 
    	    logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
    	    max_prob_index = tf.argmax(logit_words, 1)[0]
    	    generated_words.append(max_prob_index)
    	    probs.append(logit_words)
	
    	    #with tf.device("/gpu:2"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
    	    current_embed = tf.expand_dims(current_embed, 0)
	 
    	    embeds.append(current_embed)
	 
    	return video, video_mask, generated_words, probs, embeds
 

    def build_model(self):

        # for every video in the batch(50), there are n_video_lstm_step(80) represented by a vector of length 1000
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image], name = "video")

        # 1 - for video input and  0 - for no video input 
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step], name = "video_mask")

        #  placeholder that holds the captions
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step + 1], name = "caption")

        # caption word present - 1 not present - 0
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step + 1], name = "caption_mask")

        # flatten the video placeholder shape(50,80,4096) to (4000,4096) shape
        video_flat = tf.reshape(video, [-1, self.dim_image])

        # do the matrix multiplication operation and addition of biases
        # encode_image_W has dimension = (4096,1000)
        # encode_image_b has dimension = (1000)
        # video_flat has shape = (4000, 4096)
        # obtained dimension = (4000, 1000)
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        # reshape from (4000, 1000) back to (50, 80, 1000)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0
	lbls = []
	predictions = []
        # encoding phase
        for i in range(0, self.n_video_lstm_step):
	    if i>0:
                tf.get_variable_scope().reuse_variables()

            # get the state (50,2000) and output(50,1000) from the lstm1 and use it over the timestpes
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
                # As per the paper zeroes are padded to the output of the lstm1 and the fed into the lstm2
                # dimension of output1 = (50, 1000) for ith step
                # dimension of padding = (50, 1000)
                # after concatenation dimension becomes = (50, 2000)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

                # output2 dimension = (50, 1000) for ith step

        # decoding step
        print "---- decoding ----"
        for i in range(0, self.n_caption_lstm_step):
            #with tf.device("/gpu:2"):
                # looks up the embedding for all the words of all the batches for the current lstm step
	    tf.get_variable_scope().reuse_variables()

            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            # for the ith timestep get all the caption placeholders
            # labels = tensor of shape (50,1)
            labels = tf.expand_dims(caption[:, i + 1], 1)
            # generate an indexing from 0 to batchsize-1
            # tf.range(start, limit, delta) just like np.arange()
            # labels = tensor of shape (50,1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)

            # concat both these to get a tensor of shape (50,2)
            # concated stores the complete index where 1 should be placed, on all other places 0s are placed
            concated = tf.concat([indices, labels], 1)

            # onehot encoding for the words - dimension is (50, vocabulary)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocabulary]), 1.0, 0.0)

            # logit_words has dimension (50, vocabulary)
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)

            # calculate the cross-entropy loss of the logits with the actual labels
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)

            # find cross_entropy loss only where mask = 1
            cross_entropy = cross_entropy * caption_mask[:, i]

            # store the probabilities
            probs.append(logit_words)
	    lbls.append(onehot_labels)
            current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            loss = loss + current_loss
	    predictions.append(tf.nn.softmax(logit_words)) 
        return loss, video, video_mask, caption, caption_mask, probs, predictions, lbls

#=====================================================================================
# Global Parameters as in the paper
#=====================================================================================
video_path = './VideoDataset'

video_train_feat_path = './Features/'
video_test_feat_path = './TestFeatures/'

video_train_data_path = './video_corpus.csv'
video_test_data_path = './test_video_list.csv'

model_path = './models/'

#=======================================================================================
# Train Parameters as in the paper
#=======================================================================================
dim_image = 4096
dim_hidden= 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 1000
batch_size = 50
learning_rate = 0.0001

# this function takes as input the csv file path and the video feature directory path and returns a dataframe that has
# both the captions and the feature path for each video
def get_video_train_data(video_data_path, video_feat_path):
    # read the csv
    video_data = pd.read_csv(video_data_path, sep=',')

    # get all the rows with english captions
    video_data = video_data[video_data['Language'] == 'English']

    # add a new column for video path
    video_data['video_path'] = video_data.apply(
        lambda row: row['VideoID'] + '_' + str(int(row['Start'])) + '_' + str(int(row['End'])) + '.avi.npy', axis=1)

    # for every video complete its video path by adding prefix of video_feat_path to to the video name
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))

    # find all the rows for which the path exists
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(x))]

    # find all the rows for which the description is string
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    # find all unique paths
    unique_filenames = sorted(video_data['video_path'].unique())

    # get all the rows with distinct video paths - this is our train data
    train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]

    return train_data


def get_video_test_data(video_data_path, video_feat_path):
    video_data = pd.read_csv(video_data_path, sep=',')
    #video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    #video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return test_data


# this function takes the sentences and returns the idsToWords and wordsToIds dictionaries
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)

    word_counts = {}
    nsents = 0
    # iterate over all sentences
    for sent in sentence_iterator:
        nsents += 1
        # iterate over all the words of every sentence to update word counts
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    # make vocabulary of all the words that have word count greater than threshold 
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    # create array for idToWords
    idtoword = {}
    idtoword[0] = '<pad>'
    idtoword[1] = '<bos>'
    idtoword[2] = '<eos>'
    idtoword[3] = '<unk>'

    wordtoid = {}
    wordtoid['<pad>'] = 0
    wordtoid['<bos>'] = 1
    wordtoid['<eos>'] = 2
    wordtoid['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoid[w] = idx+4
        idtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    # dimension = size of vocabulary
    bias_init_vector = np.array([1.0 * word_counts[ idtoword[i] ] for i in idtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoid, idtoword, bias_init_vector

def accuracy(predictions, labels): 
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
    accu = (100.0 * correctly_predicted) / predictions.shape[0] 
    return accu 

def train():
    train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
    train_captions = train_data['Description'].values
    #test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    #test_captions = test_data['Description'].values
    captions_list = list(train_captions) #+ list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)
    wordtoid, idtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    np.save("./data/wordtoid", wordtoid)
    np.save("./data/idtoword", idtoword)
    np.save("./data/bias_init_vector", bias_init_vector)

    model = V2D(
        dim_image=dim_image,
        vocabulary=len(wordtoid),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_lstm_steps=n_frame_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=bias_init_vector)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "5"
    #config.gpu_options.per_process_gpu_memory_fraction=0.90
    config.gpu_options.allow_growth=True
    with tf.variable_scope(tf.get_variable_scope()) as scope:   
        tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs, tf_predictions, tf_lbls = model.build_model()
        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    print "===================================== Model Built ====================================="
    sess = tf.InteractiveSession(config = config)
    saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2)
    with tf.variable_scope(tf.get_variable_scope()):
    	train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    #loss_fd = open('loss.txt', 'w')
    #loss_to_draw = []
    #current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    #print current_train_data
    for epoch in range(0, n_epochs):
        print "===================================== Epoch "+str(epoch)+" ====================================="
#	loss_to_draw_epoch = []

        # reshuffle the train data using "pandas.dataframe.index to create index over the dataset" and "pandas.dataframe.ix
        # method to get the rows of the dataframe given the index"
        # print train_data
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]
	
        # group all the videos with the same captions then select make a uniform random choice of one caption from all
        # the captions in the group for that video
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        
        # index the dataframe
        # current_train_data = current_train_data.reset_index(drop=True)

        for start, end in zip(range(0, len(current_train_data), batch_size),
                              range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()
            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values
	    '''with open("video_list.txt",'w') as f:	    
		for v in current_videos:
                    f.write(v+"\n")'''
            
            # initialize the current_feats for all the videos in the batch size
            # i.e. for every frame a vector of size dim_image
            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))

            current_feats_vals = map(lambda feature: np.load(feature), current_videos)

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1
       
            current_captions = current_batch['Description'].values
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)

            # iterate over the entire batch of captions and make them of length <= n_caption_lstm_steps
            # also append <eos> at the end of the caption
            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    # get all the words except the last
                    # the last is <eos>
                    for i in range(n_caption_lstm_step - 1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            captions = []
            for cap in current_captions:
                caption_of_ids = []
                for word in cap.lower().split(' '):
                    if word in wordtoid:
                        caption_of_ids.append(wordtoid[word])
                    else:
                        caption_of_ids.append(wordtoid['<unk>'])
                captions.append(caption_of_ids)
            # convert from list to matrix of dimension (batchsize, n_caption_lstm_steps)
            # using keras.preprocessing.sequence.pad_sequences to pad the captions of length smaller than
            # n_caption_lstm_steps

            current_caption_matrix = sequence.pad_sequences(captions, padding='post',
                                                            maxlen=n_caption_lstm_step)
            # add a zero at the end of each caption
            current_caption_matrix = np.hstack(
                [current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            # count the number of non-zeros in each row
            # 0 is  for padding
            # one additional 0 is at the end
            # therefore count all padding zeros and the last 0
            nonzeros = np.array(map(lambda x: (x != 0).sum() + 1, current_caption_matrix))

            # create the caption mask  i.e. place a 1 if there is a word and 0 if there is a <pad>
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            probs_val = sess.run([tf_probs], feed_dict={
                tf_video: current_feats,
                tf_caption: current_caption_matrix
            })
	    '''print np.array(onehot_labels_val).shape
	    for x in onehot_labels_val[0]:
		for y in x:
		    for i,z in enumerate(y):
		        if z == 1:
			    print idtoword[i]'''
				 
            _, loss_val, predictions, lbls = sess.run(
                [train_op, tf_loss, tf_predictions, tf_lbls],
                feed_dict={
                    tf_video: current_feats,
                    tf_video_mask: current_video_masks,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks
                })
	    '''cps = []
	    act = []
            for i in range(50):
	        cps.append([])
		act.append([])
	    print np.array(lbls).shape
	    print np.array(predictions).shape'''
            mean_accuracy = 0.0
            c = 0.0
	    for p,l in zip(predictions,lbls):
		#print("Epoch accuracy on train Data: {:.1f}%".format(accuracy(p,l))) 
		mean_accuracy+=accuracy(p,l)
                c+=1.0
		'''for i in range(50):	
		    cps[i].append(np.argmax(p[i], 0))
		    act[i].append(np.argmax(l[i], 0))
            for i in range(50):
	        print "True Caption"
		print_caption(act[i],idtoword)
		print "Predicted Caption"
		print_caption(cps[i],idtoword)'''	
            mean_accuracy /= c
         #   loss_to_draw_epoch.append(loss_val)

            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, "accuracy: ", mean_accuracy,' Elapsed time: ', str(
                (time.time() - start_time))

        if np.mod(epoch,10)==0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)    
        #    loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + )
'''	loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_dir = "./loss_imgs"
        plt_save_img_name = "loss_"+str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name)) 
        if np.mod(epoch, 10) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
'''
    #loss_fd.close()
def print_caption(caption, idtoword):
    cp = ""
    for idx in caption:
	cp+=idtoword[idx]+" "
    print cp

def test(model_path='./models/'):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    print test_data
    test_videos = test_data['video_path'].unique()
    test_video_names = test_data['VideoID'].unique()
    idtoword = pd.Series(np.load('./data/idtoword.npy').tolist())
    tf.reset_default_graph()
    bias_init_vector = np.load('./data/bias_init_vector.npy')
    model = V2D(
        dim_image=dim_image,
        vocabulary=len(idtoword),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_lstm_steps=n_frame_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        #isTrain = True,
        bias_init_vector=bias_init_vector)
    #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name):
     #   print i 

    '''config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "3"
    #config.gpu_options.per_process_gpu_memory_fraction=0.90
    config.gpu_options.allow_growth=True
    saver = tf.train.import_meta_graph(model_path)
    with tf.Session(config=config).as_default() as sess:
    #sess.run(tf.global_variables_initializer())        
    	#saver = tf.train.import_meta_graph(model_path)
    	saver.restore(sess, tf.train.latest_checkpoint('./models1/'))
    	print Wemb.eval()'''
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    saver = tf.train.Saver()

    #sess = tf.InteractiveSession()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    #config.gpu_options.per_process_gpu_memory_fraction=0.90
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())	
    saver.restore(sess, os.path.join(model_path, 'model-990'))
    #print(sess.run(model.Wemb))
    test_output_txt_fd = open('S2VT_results3.txt', 'w')
    test_actual_output_txt_fd = open('actual_captions.txt','w')
    for idx, row in test_data.iterrows():
	test_actual_output_txt_fd.write(str(row['VideoID'])+"\t")
	test_actual_output_txt_fd.write(str(row['caption'])+"\n")
    for idx, video_feat_path in enumerate(test_videos):
        print idx, video_feat_path
        #video_feat_path = '/home/shashwatcs15/VideoAnalytics/La3qC1Y6cX4_9_30.avi.npy'
        video_feat = np.load(video_feat_path)[None, ...]
        # video_feat = np.load(video_feat_path)
        # video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
	#print(sess.run(model.embed_word_W))
        print generated_word_index
        generated_words = idtoword[generated_word_index]

        #punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        #generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
	generated_sentence = generated_sentence.replace(' <pad>', '')
        print generated_sentence, '\n'
        test_output_txt_fd.write(test_video_names[idx] + '\t')
        test_output_txt_fd.write(generated_sentence + '\n')
        #break

if __name__=='__main__':
	#create_vocab()
	#os.environ["CUDA_VISIBLE_DEVICES"]="2"
	test()
        #train()
