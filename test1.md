
Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

# Attention Is All You Need

## Abstract

The dominant sequence-to-sequence translation models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models are based on the Transformer architecture, which is inspired by the attention mechanism. We propose a new simple network translation architecture, the Transformer, based on self-attention. The Transformer is a completely different approach that replaces the entire encoder-decoder architecture with self-attention. Experiments on two machine translation tasks show these models to be superior to the best performing models in the literature, and significantly less to train. Our model achieves 28.4 BLEU on the WMT 2014 English-German translation task, which is 1.4 BLEU better than the best performing model, and 1.6 BLEU training for 3.5 days on eight GPUs, a small fraction of the training costs of the best performing model. The model is also significantly faster to train, as well as to other tasks by applying it successfully to English constituency parsing both with large and limited training data.




# 1 Introduction

Recurrent neural networks, long short-term memory (LSTM) and gated recurrent neural networks in particular, have been firmly established as state-of-the-art approaches in sequence modeling and translation. However, these models are limited by the number of operations required to relate signals from two arbitrary input or output positions. This inherently requires a large number of parameters, which is a problem for long sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in the efficiency of sequential computation, while also improving model performance in case of the latter. The fundamental contribution of this work is the Transformer.

Attention mechanisms have become an integral part of competing sequence modeling and translation models in various tasks. The modeling of dependencies without regard to their distance in the input or output sequence is a key challenge. The Transformer is the first translation model that uses attention in conjunction with a recurrent network.

In this section, we will describe the Transformer, an auto-regressive architecture exchanging recurrence and initial relying entirely on an attention mechanism to draw long dependencies between input and output. The Transformer is also the first translation model that uses self-attention, which is the state-of-the-art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU (ENN) [5], ByteNet [6] and ConvS2S [7], all of which use convolutional neural networks as basic building blocks. However, these models are limited by the number of operations required to relate signals from two arbitrary input or output positions. This inherently requires a large number of parameters, which is a problem for long sequence lengths, as memory constraints limit batching across examples. The Transformer is the first translation model that uses attention in conjunction with a recurrent network.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used in various tasks, including language modeling, machine translation, and image captioning. It has been found to be effective in capturing long-range dependencies in sequential data. The Transformer is the first translation model that uses self-attention, which is the state-of-the-art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# 3 Model Architecture

Most competitive neural sequence translation models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbols representing a sentence, (x1, ..., xn), to a sequence of continuous representations, (h1, ..., hn). The decoder then generates an output sequence of symbols (y1, ..., yn) or symbols one element at a time. At each step the model is auto-regressive (i.e., consuming the previously generated symbols as additional input when generating the next).



Figure 1: The Transformer - model architecture.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.

### 3.1 Encoder and Decoder Stacks

**Encoder:**
The encoder is composed of a stack of \( V = 6 \) identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. The output of the sub-layers is then normalized and added to the input of the sub-layer. The two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer could be either an attention sub-layer or a fully connected feed-forward sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce output vectors of dimension \( d_{\text{model}} = 512 \).

**Decoder:**
The decoder is also composed of a stack of \( V = 6 \) identical layers. In addition to the two sub-layers in each decoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder. This allows the decoder to use information from the encoder, as well as information from the previous decoder layers. Each sub-layer is an additive sub-layer. The output of the sub-layers is then normalized and added to the input of the sub-layer. The two sub-layers, followed by layer normalization [1]. We can modify the self-attention sub-layers so that they attend to a different set of positions. In particular, the decoder attentions do not attend to the future positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position \( i \) can depend only on the known outputs at positions less than \( i \).

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum



Scaled-Dot Product Attention

We can also consider the attention mechanism as a scaled dot-product attention (Scaled-Dot-Product Attention) (Figure 2). The input consists of queries and keys of dimension $d_k$ and values of dimension $d_v$. We compute the dot products of the queries with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$, keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The two most commonly used attention functions are additive attention and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor $\frac{1}{\sqrt{d_k}}$ and the use of a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much more computationally efficient in practice, as it can be implemented using highly optimized matrix multiplication code.

While for small values of $d_k$, the two mechanisms perform similarly, additive attention outperforms dot-product attention. For large values of $d_k$, dot-product attention performs better. For example, for $d_k = 64$, the dot products grow in magnitude, pushing the softmax function into regions where it has extreme gradients, which can cause numerical issues. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

Multi-Head Attention

Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values, and queries, we found it beneficial to linearly project the queries, keys, and values $h$ times with different, learned linear transformations. We then compute the dot products of the $h$-dimensional queries, keys, and values and we then perform the attention function in parallel, yielding $h$-dimensional outputs.

To motivate why the dot products get large, assume that the components of $Q$ and $K$ are independent random variables with mean 0 and variance 1. Then their dot products, $q_k \cdot k_k = \sum_{i=1}^{d_k} q_{ki}k_{ki}$, has mean 0 and variance $d_k$.



output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure\ref{fig:2}.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QK^i, KV^i), W^O â R^{d_{out}Ãd_{out}}

When the projections are parameter matrices W^Q â R^{d_{in}Ãd_{in}}, W^K â R^{d_{in}Ãd_{in}}, W^V â R^{d_{in}Ãd_{in}}, and W^O â R^{d_{out}Ãd_{out}}.

In this work we employ h = 6 parallel attention layers, or heads. For each of these we use d_h = d_in / d_head = 64. Due to the reduced dimension of each head, the total computational cost is much lower than that of a single fully connected layer of the same dimensionality.

3.2.3 Application of Attention in our Model

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the output of the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend to all positions in the encoder. This is a typical encoder-decoder attention mechanism in sequence-to-sequence models such as \cite{DBLP:journals/corr/ChoMGBBSB14}.

- The encoder contains self-attention layers. In a self-attention layer all of the keys, values, and queries come from the same position in the encoder. Each position in the encoder can attend to all positions in the encoder. This is a typical self-attention mechanism in transformer models.

- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the previous layer of the decoder and to including that of the encoder. This is a typical self-attention mechanism in transformer models.

3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation function in between.

F(Nz) = (m(1)W1 + b1)z + (m(2)W2 + b2)z

While the linear transformations are different in each layer, they use different parameters, they differ from layer to layer. Another way of describing this is two convolutions with kernel size 1. The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionally d_ff = 2048.

3.4 Embeddings and Softmaxs

Similarly to other sequence translation models, we use learned embeddings to convert the input tokens into vectors. We also use learned linear transformations and softmax functions to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrices between the two embedding layers and the pre-softmax linear transformation, similar to \cite{DBLP:journals/corr/ChoMGBBSB14}. In the embedding layers, we multiply these weights by ï¿½d_model.

5



# 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. This information is usually added to the inputs of the model, at the bottoms of the encoder and decoder stacks. The positional encoding has the same dimension $d_{\text{max}}$ as the input embedding, so that the two can be summed. There are many choices of positional encodings, learned and fixed (E).

In this work, we use one and cosine functions of different frequencies:

$P_{\text{Enc}(2i)} = \sin(i/1000^{2i/d_{\text{max}}})$

$P_{\text{Enc}(2i+1)} = \cos(i/1000^{2i/d_{\text{max}}})$

where $i$ is the position and $d$ is the dimension. This is each dimension of the positional encoding corresponds to a fixed frequency. The frequencies range from $2i$ to $100000 \cdot 2i$: We chose this function because we hypothesized it would allow the model to easily learn to attend by relative position rather than absolute position (which may be difficult).

We also experimented with a learned positional encoding (E), and found that the two versions produced nearly identical results (see Table (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention

In this section we compare various types of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations ($x_1, ..., x_n$) to another variable-length sequence of symbol representations ($y_1, ..., y_m$). A self-attention layer in a typical sequence-to-sequence encoder or decoder. Motivating our use of self-attention we consider the following example:

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The self-attention layer is able to capture long-range dependencies, which is a key challenge in many sequence-to-sequence tasks. One key factor affecting the ability of a model to capture long-range dependencies is the maximum path length between any two input and output positions in the network. The shorter these paths between any combination of positions in the input and output sequences, the more likely the model is to be able to capture long-range dependencies. The maximum path length between any two input and output positions in networks composed of the different types of layers is shown in Table (E).

As noted in Table (E), a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence



# 5 Training

This section describes the training regime for our models.

## 5.1 Training Data and Batchsize

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. We used byte-pair encoding (BPE) which has a shared source-target vocabulary of about 7000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset, which has a shared source-target vocabulary of about 100,000 tokens. Sentence pairs were binned together by approximate sentence length. Each training batch contained a set of sentence pairs containing approximately 2500 source tokens and 2500 target tokens.

## 5.2 Hardware and Schedule

We trained our models on a machine with 8 NVIDIA TPU v3 GPUs. For our base models using the byte-pair encoding, the training throughput for the paper (each training step took about 0.4 seconds. We trained the base model for a total of 10 billion steps or 12 hours. For our big models (described on the paper), we trained for 100 billion steps or 1.1 trillion steps, which took about 3.5 days.

## 5.3 Optimizer

We used the Adam optimizer with Î²1 = 0.5, Î²2 = 0.98 and Îµ = 10â9. We varied the learning rate over the course of training, according to the formula:


lr_t = Î± * tâ0.5, if step_num < warmup_steps * tâ1/2


This corresponds to decreasing the learning rate linearly for the first warmup_steps training steps, and decreasing it quadratically proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

## 5.4 Regularization

We employ three types of regularization during training:

- **Length normalization**: The input sequence is centered around the output position. This would increase the maximum path length to O(n + k) in the separable convolutional approach.
- **Input position bias**: A single convolutional layer with a kernel width k = 1 does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(n/k) separable convolutional layers in the case of non-contiguous kernels. The complexity of connecting between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, and the complexity of a separable convolutional layer is O(kn + n + d2). With k = n, however, the complexity of a separable convolutional layer is O(n + d2), which is considerably less than the complexity of a convolutional layer, O(kn + n + d2).
- **Output position bias**: The approach we take in our model.



Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French WMT14 test sets at a fraction of the training cost.

| Model | BLEU | Training Cost (FLOPs) |
|-------|------|----------------------|
| EN-DE | EN-FR | EN-DE | EN-FR |
| BERT (No) | 27.3 | 39.2 | 1.0 | 1.0 |
| Deep-Att + Pool-Volk (E) | 24.6 | 40.6 | 2.3 | 1.0 |
| CNTT (E) | 25.1 | 40.6 | 2.3 | 1.0 |
| Cores2S (E) | 25.16 | 40.46 | 9.0 | 10.0 |
| M4 (E) | 26.03 | 40.46 | 1.5 | 10.0 |
| Deep-Att + Pool-Volk Ensemble (E) | 26.30 | 40.46 | 2.3 | 1.0 |
| CNTT Ensemble (E) | 26.30 | 40.46 | 2.3 | 1.0 |
| Cores2S Ensemble (E) | 26.36 | 41.29 | 2.7 | 10.0 |
| Transformer (base model) | 27.7 | 38.7 | 3.3 | 1.0 |
| Transformer (big) | 29.4 | 41.8 | 2.3 | 1.0 |

Residual Dropout: We apply dropout (E) to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings, with a probability of 0.1. The probability of dropout is 0.1.

Label Smoothing: We apply label smoothing as the model learns to be more sure, but improves accuracy and BLEU score. The input to the loss function is as follows: \( y_{true} = 1 - \epsilon \), \( y_{pred} = \epsilon / V \). This configuration of this model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the other configurations.

6 Results

6.1 Machine Translation

6.1.1 English-to-German Translation

On the WMT'2014 English-to-German translation task, our big transformer model (Transformer (big)) in Table 2 outperforms the best previously reported models (including ensembles) by more than 2.1 BLEU, establishing a new state-of-the-art BLEU score of 29.4. The configuration of this model is based on the Transformer (big) model, but with a larger hidden size of 1024 and 16 attention heads. This surpasses all previously published models and ensembles, at a fraction of the training cost of any of the other configurations.

On the WMT'2014 English-to-French translation task, our big model achieves a BLEU score of 41.1, surpassing all previously published models and ensembles, at a fraction of the training cost of any of the previous state-of-the-art models. The Transformer (big) model trained for English-to-French used dropout (E) on the output of each sub-layer, before it is added to the sub-layer input and normalized.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used the same training schedule as the base models, but with a larger hidden size of 1024 and 16 attention heads. These values were chosen after experimentation on different configurations.

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures. We also provide a breakdown of the training cost in terms of FLOPs, which is the number of floating-point operations required to train the model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.


Table 3. Variations on the Transformer architecture. Undlined values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

| N | d_model | N_enc | d_ff | N_dec | d_ff | P_enc | P_dec | train steps | (PPL) (dev) | BLEU (dev) | t (10^6) |
|---|---------|-------|------|-------|------|-------|-------|------------|------------|-----------|----------|
| base | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 0.01 | 490K | 4.92 | 25.8 | 65 |
| (A) | 128 | 128 | 16 | 32 | 32 | 0.1 | 0.1 | 0.01 | 100K | 4.92 | 25.8 | 65 |
| (A) | 4 | 128 | 128 | 16 | 32 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 42.5 | 50 |
| (A) | 16 | 128 | 32 | 32 | 16 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 42.5 | 50 |
| (B) | 4 | 512 | 16 | 32 | 32 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (B) | 2 | 512 | 16 | 32 | 32 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (B) | 4 | 512 | 16 | 32 | 32 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (C) | 256 | 32 | 32 | 1024 | 1024 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (C) | 1024 | 32 | 128 | 128 | 1024 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (C) | 4096 | 32 | 128 | 1024 | 1024 | 0.1 | 0.1 | 0.01 | 500K | 2.55 | 58 | 50 |
| (D) | 1 | 512 | 16 | 32 | 32 | 0.0 | 0.0 | 0.0 | 500K | 5.77 | 24.6 | 50 |
| (D) | 2 | 512 | 16 | 32 | 32 | 0.0 | 0.0 | 0.0 | 500K | 4.67 | 25.7 | 50 |
| (E) | big | 1024 | 4096 | 16 | 32 | 0.3 | 0.0 | 0.0 | 300K | 4.92 | 25.7 | 213 |

development set, newstest2013. We present these results in Table 5.

In Tables 3 (A), we vary the number of attention heads and the attention size and value dimensions, keeping the number of layers constant. We also vary the number of layers in Section 3.2. We find that increasing the number of attention heads to 0.9 PPL is worse than the best setting, quality also drops off with too many heads. In Table 3 (B), we vary the number of attention heads and the attention size and value dimensions, keeping the number of layers constant. We find that increasing the number of attention heads to 0.9 PPL is worse than the best setting, quality also drops off with too many heads. This suggests that determining compatibility is not easy and that a more sophisticated compatibility measure is needed. In Table 3 (C), we vary the number of layers and the number of attention heads. We find that bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (D) we replace our standard positional embedding with learned positional embeddings [2] and conserve nearly identical results to the base model.

6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strict structural constraints, and the input is a sequence of words. Previous work on constituency parsing with RNN-based models have not been able to attain state-of-the-art results in small data regimes [23].

We performed experiments on the Penn Treebank [23] about 40k training sentences. We also trained it in a semi-supervised setting, using the 10k training sentences from the Penn Treebank and the 10k training sentences from the CoNLL-2009 shared task [27]. We used a vocabulary of 58k tokens for the WSD only setting and a vocabulary of 32k tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual connections, learning rates and beam size on the Section 2.2 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we



Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)

| Paper | Training | WSJ 23(F1) |
|-------|----------|------------|
| Vinyals & Le (2015) [20] | 24 | 90.4 |
| Povey et al. (2006) [2] | WSJ only, discriminative | 90.4 |
| Povey et al. (2006) [2] | WSJ only, semi-supervised | 90.4 |
| Dyer et al. (2016) [8] | WSJ only, discriminative | 91.7 |
| Dyer et al. (2016) [8] | WSJ only, semi-supervised | 91.7 |
| Zhu et al. (2017) [31] | semi-supervised | 91.3 |
| He et al. (2017) [11] | semi-supervised | 91.3 |
| McCleary et al. (2006) [26] | semi-supervised | 92.1 |
| Vaswani et al. (2017) [32] | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Transformer (4 layers) | generative | 93.0 |
| Dyer et al. (2016) [8] | generative | 93.3 |

increased the maximum output length in input length >= 300. We used a beam size of 21 and Î± = 0.3 for both WSJ and the semi-supervised setting.

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well. The Transformer outperforms all previous reported ensembles on the WSJ dataset, with the exception of the Recurrent Neural Network Grammar [3].

In contrast to previous sequence-to-sequence models [2], the Transformer outperforms the BerkeleyParser [2] even when training only on the WSJ training set of 40k sentences.

7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both the 2014 English-to-English and English-to-French WMT datasets, the Transformer is a new state-of-the-art. In the former task our best model outperforms all previously reported ensembles.

We are currently working on extending the Transformer to other tasks. We have a plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to more complex tasks such as image captioning, video captioning, and image generation.

The code used to train and evaluate our models is available at <https://github.com/ tensorflow/text2seq>.

Acknowledgements: We are grateful to Nat Kakhkabemmer and Stephen Gouws for their fruitful comments, corrections and inspiration.

References

[1] Eugene Bi, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Shuming Gong, Li Dong, and Xiaodong Liu. Long short-term memory networks for machine reading. arXiv preprint arXiv:1702.00733, 2016.



[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[7] Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, Denis Yarats, and Noah S. Smith. Recent neural network models for sequence tagging. arXiv preprint arXiv:1705.03122, 2017.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770â778, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Pascal Frasconi, and JÃ¼rgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. arXiv preprint arXiv:1206.5538, 2012.

[13] Sepp Hochreiter and JÃ¼rgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735â1780, 1997.

[14] Zongqiang Huang and Mary Harper. Soft-training PFGM-grammars with latent annotations. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832â841. ACL, August 2009.

[15] Kaili Huang, Xiaodong Liu, Xiaodong Liu, Mike Carbin, Xiaodong Liu, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02201, 2016.

[16] Lukasz Kaiser and Surya Bhat. Can active memory replace attention? In Advances in Neural Information Processing Systems, pages 2025â2033, 2017.

[17] Lukasz Kaiser and Ilya Sutskever. Neural OPU-learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[18] Nat Kalchbrenner, Lasse Espeholt, Karen Simonyan, Amir Rusu, and Oriol Vinyals. Key-Value Networks for efficient attention in linear time. arXiv preprint arXiv:1610.00999, 2017.

[19] Yoon Kim, Carl Doersch, Bingbing Huang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1603.04938, 2016.

[22] Zhihuan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[23] Minh-Thang Luong, Hoai Le, Ilya Sutskever, Oriol Vinyals, and Kazuho Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1711.02114, 2017.

[24] Minh-Thang Luong, Hoai Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.



[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: the penn treebank. Computational linguistics, 18(2):313â330, 1992.

[26] David McClosky, Eugene Charniak, and Jack Johnson. Effective self-training for parsing. In Proceedings of the 44th Annual Meeting of the Association for Computational Linguistics (ACL), pages 152â159. ACL, June 2006.

[27] Askar Parikh, Oscar TÃ¤ckstrÃ¶m, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pages 1542â1552, 2016.

[28] Roman Parshik, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04834, 2017.

[29] Slav Petrov, Leon Barrett, Romain Tibaux, and Dan Klein. Learning accurate, compact, and efficient representations of words. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433â440. ACL, July 2006.

[30] Ofer Porco and Law Wulfe. Using the output embedding to improve language models. arXiv preprint arXiv:1608.00539, 2016.

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with multi-lingual subword units. arXiv preprint arXiv:1511.08653, 2015.

[32] Naman Shin, David M. Blei, Michael I. Jordan, Matthew D. Hoffman, Andy D. Lee, Quoc V. Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06548, 2017.

[33] Nikolaos Simonyan, Andrew Zisserman, Yann LeCun, Yann LeCun, Yann LeCun, and R. Garnett. et al. Advances in Neural Information Processing Systems 28, pages 2449â2448. Curran Associates, Inc., 2015.

[34] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104â3112, 2014.

[35] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Sergey Ioffe, Justin Shlens, and Zhifeng Wu. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Advances in Neural Information Processing Systems, pages 3271â3279, 2015.

[36] Vinyals & Kaiser, Koz, Peter, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.

[37] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Shengdong Chen, Quoc V Le, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norou


Attention Visualizations

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb "making", completing the phrase "making more difficult". Attention shown here only for the word "making". Different colors represent different heads (blue = head 0, red = head 1, etc.).



Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Pull attentions for head 5. Bottom: Isolated attentions from just the word 'it' for attention heads 5 and 6. Note that the attentions are very sharp for this word.



Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 3 of 6. The heads clearly learned to perform different tasks.
