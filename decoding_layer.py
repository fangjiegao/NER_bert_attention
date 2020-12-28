import tensorflow as tf


def decoding_layer(target_letter_to_int, decoding_embedding_size,num_layers,rnn_size,
                   target_sequence_length, max_target_sequence_length,encoder_state,decoder_input):
    '''
    构造Decoder层

    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''

    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings,decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = tf.layers.Dense(target_vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))


    # 4. Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_embed_input,
                                                            sequence_length = target_sequence_length,
                                                            time_major = False)


        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,training_helper,encoder_state,output_layer)
        training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                        maximum_iterations = max_target_sequence_length)


    # 5. Predicting decoder
    # 与training共享参数

    with tf.variable_scope("decode",reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype=tf.int32),[batch_size],name='start_token')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,start_tokens,target_letter_to_int['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,impute_finished = True,
                                                                          maximum_iterations = max_target_sequence_length)


    return training_decoder_output,predicting_decoder_output