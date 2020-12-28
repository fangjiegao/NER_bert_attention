# -*- coding: UTF-8 -*-

import tensorflow as tf

'''
用注意力机制包装的Seq2Seq模型
'''

# 定义相关参数
# LSTM的隐藏层规模
HIDDEN_SIZE = 1024
# LSTM的层数
NUM_LAYERS = 2
# 英文词汇表的大小
SRC_VOCAB_SIZE = 10000
# 中文词汇表的大小
TRG_VOCAB_SIZE = 4000
# dropout层保留的概率
KEEP_PROB = 0.8
# 控制梯度膨胀的梯度大小上限
MAX_GRAD_NORM = 5
# 共享softmax层和词向量层之间的参数
SHARE_EMB_AND_SOFTMAX = True
# 句首和句末的标志
SOS_ID = 1
EOS_ID = 2


class NMTModelWithAttention(object):
    def __init__(self):

        # 构造双向循环的网络结构
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        # 定义解码器
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        )

        # 定义两种语言的词向量层
        self.src_embedding = tf.get_variable("src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 如果共享参数则用词向量层的参数初始化softmax层的参数
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        # 初始化softmax的bias参数
        self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        # 获取batch的大小
        batch_size = tf.shape(src_input)[0]

        # 分别将输入和输出转化为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 对输入和输出的词向量进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 构造编码器，使用bidirectional_dynamic_rnn构造双向循环网络
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, dtype=tf.float32
            )
            # 将两个LSTM的输出拼接为一个张量
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        # 构造解码器
        with tf.variable_scope("decoder"):
            # 选择注意力权重的计算模型
            attention_machanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs, memory_sequence_length=src_size
            )
            # 将原来的dec_cell和注意力一起封装成更高层的attention_cell
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_machanism, attention_layer_size=HIDDEN_SIZE
            )
            # 用dynamic和attention_cell来构造编码器
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # 计算解码器每一步的log perplexity
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 计算平均损失，注意在计算的时候需要将填充的位置设定为0
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 获取可训练的参数
        trainable_variables = tf.trainable_variables()

        # 定义反向传播操作、优化步骤和训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        # 返回平均损失和训练步骤
        return cost_per_token, train_op

    def inference(self, src_input):
        # 整理输入，将其转化为一个batch大小为1的batch，并转化为对应的词向量
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 定义编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, dtype=tf.float32
            )
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        # 定义解码器
        with tf.variable_scope("decoder"):
            attention_machanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs, memory_sequence_length=src_size
            )
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_machanism, attention_layer_size=HIDDEN_SIZE
            )

        # 定义解码的最大步骤避免无限循环
        MAX_DEC_LEN = 100

        # 另外定义解码过程
        with tf.variable_scope("decoder/rnn/attention_wrapper"):
            # 初始化生成的句子，用SOS_ID开头
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            init_array= init_array.write(0, SOS_ID)
            # 初始循环状态以及循环步数
            init_loop_var = (attention_cell.zero_state(batch_size=1, dtype=tf.float32), init_array, 0)

            # 定义循环结束条件
            def continue_loop_condition(state, trg_ids, step):
                # 如果遍历到了句子结尾或者比定义的最大步骤大就停止循环
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN - 1)
                ))

            # 定义循环体
            def loop_body(state,trg_ids, step):
                # 读取最后一步输出的单词并读取其对应的词向量
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                # 调用包装好的attention_cell计算下一步的结果
                dec_outputs, next_state = attention_cell.call(state=state, inputs=trg_emb)
                # 计算结果对应的softmax层结果
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                # 根据logits结果求得可能性最大的单词
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将当前计算出来的单词结果写入trg_ids
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            # 调用tf.while_loop，返回最终结果即trg_ids
            state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
            # 返回trg_ids里的值
            return trg_ids.stack()