# 5. Natural Language Processing With RNNs

###### tags: `FreeCodeCamp-MLwithPython`


* [Sequence Data](#sequence-data)
    * [Ecoding Text](#encoding-text)
    * [Bag of Words](#bag-of-words)
    * [Integer Encoding](#integer-encoding)
    * [Word Embeddings](#word-embeddings)
* [Recurrent Neural Networks (RNN's)](#recurrent-neural-networks)
    * [LSTM](#lstm)
* [Sentiment Analysis](#layer-parameters)
* [Sources](#sources)

---

Natural Language Processing (or NLP for short) is a discipline in computing that deals with the communication between natural (human) languages and computer languages. A common example of NLP is something like spellcheck or autocomplete. Essentially NLP is the field that focuses on how computers can understand and/or process natural/human languages. 

---

## Sequence Data

In this case we will look at sequences of text and learn how we can encode them in a meaningful way. Unlike images, sequence data such as long chains of text, weather patterns, videos and really anything where the notion of a step or time is relevant needs to be processed and handled in a special way.

But what do I mean by sequences and why is text data a sequence? Since textual data contains many words that follow in a very specific and meaningful order, we need to be able to keep track of each word and when it occurs in the data. 

Simply encoding say an entire paragraph of text into one data point wouldn't give us a very meaningful picture of the data and would be very difficult to do anything with. This is why we treat text as a sequence and process one word at a time. We will keep track of where each of these words appear and use that information to try to understand the meaning of peices of text.

In other workds: how we can turn some textual data into numeric data that we can feed to out neural network?

### Encoding Text

As we know machine learning models and neural networks don't take raw text data as an input. This means we must somehow encode our textual data to numeric values that our models can understand. There are many different ways of doing this.

Before we get into the different encoding/preprocessing methods let's understand the information we can get from textual data by looking at the following two movie reviews.

I thought the movie was going to be bad, but it was actually amazing!

I thought the movie was going to be amazing, but it was actually bad!

Although these two setences are very similar we know that they have very different meanings. This is because of the **ordering** of words, a very important property of textual data.

Now keep that in mind while we consider some different ways of encoding our textual data.


### Bag of Words

The first and simplest way to encode our data is to use something called bag of words. This is a pretty easy technique where each word in a sentence is encoded with an integer and thrown into a collection that does not maintain the order of the words but does keep track of the frequency. 

![](https://i.imgur.com/G800qQN.png)

### Integer Encoding

This involves representing each word or character in a sentence as a unique integer and maintaining the order of these words.

Ideally when we encode words, we would like similar words to have similar labels and different words to have very different labels. For example, the words happy and joyful should probably have very similar labels so we can determine that they are similar. While words like horrible and amazing should probably have very different labels. The method we looked at above won't be able to do something like this for us. This could mean that the model will have a very difficult time determing if two words are similar or not which could result in some pretty drastic performace impacts.

### Word Embeddings

This method keeps the order of words intact as well as encodes similar words with very similar labels. It attempts to not only encode the frequency and order of words but the meaning of those words in the sentence. It encodes each word as a dense vector that represents its context in the sentence.

Unlike the previous techniques word embeddings are learned by looking at many different training examples. You can add what's called an embedding layer to the beggining of your model and while your model trains your embedding layer will learn the correct embeddings for words. You can also use pretrained embedding layers.

In other words: Try to find a way to represent words that are similar using very similar numbers.

It does this by classify or translate every single word into a vector. This vector is gonna have n dimensions, usually 64 or 128.

Every single component of that vector would tell us what group it belongs to, or how similar is to other words.

![](https://i.imgur.com/4yxQ5nq.png)


---

## Recurrent Neural Networks

Up until this point we have been using something called **feed-forward** neural networks. This simply means that all our data is fed forwards (all at once) from left to right through the network. This was fine for the problems we considered before but won't work very well for processing text. 

After all, even we (humans) don't process text all at once. We read word by word from left to right and keep track of the current meaning of the sentence so we can understand the meaning of the next word. Well this is exaclty what a recurrent neural network is designed to do. 

When we say recurrent neural network all we really mean is **a network that contains a loop**. A RNN will process one word at a time while maintaining an **internal memory** of what it's already seen. This will allow it to treat words differently based on their order in a sentence and to slowly build an understanding of the entire input, one word at a time.

This is why we are treating our text data as a sequence.

In other words: the fundamental difference between a recurring neural network or somethig like a dense or convulotioanl NN is the fact that it contains an internal loop, what this really means is that our RNN does not process our entire data at once, it processes it in different timesteps and mantains what we call an internal memory and kind of an internal state.

Let's have a look at what a recurrent layer might look like.

![](https://i.imgur.com/I4tyhiU.png)
Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

* **ht**: output at time t.
* **xt**: input at time t.
* **A**: Recurrent Layer (loop).

What this diagram is trying to illustrate is that a recurrent layer processes words or input one at a time in a combination with the output from the previous iteration. So, as we progress further in the input sequence, we build a more complex understanding of the text as a whole.

What we've just looked at is called a **simple RNN layer**. It can be effective at processing shorter sequences of text for simple problems but has many downfalls associated with it. One of them being the fact that as text sequences get longer it gets increasingly difficult for the network to understand the text properly.

In detail:

Rather than passing all infromation at once we are goint to pass it as a sequence, which means that we are goint to pass one word at a time to this recurrent layer.

At timestep 0 the internal state of this layer is nothing, we havent seen anything yet. Which means that this first cell (0) is only going to look and consider the first word x0 (hi).

So cell 0 is gonna take that word, do some math and make a prediction about it giving an output o0.
![](https://i.imgur.com/Rp0933G.png)

So when the first cell is done computing and we have its output, we want to process the second word, and to make the computation and create the second output we will use de second word (x1) and the first output (o1). And so on:
![](https://i.imgur.com/JOs5jh8.png)
![](https://i.imgur.com/aeS8jU0.png)

The issue with this is as we have a very long sequence of words, the beggining of those sequences starts to kind of get lost as we go through the process.

### LSTM

The layer we dicussed in depth above was called a simple RNN. However, there does exist some other recurrent layers (layers that contain a loop) that work much better than a simple RNN layer. 

The one we will talk about here is called **LSTM (Long Short-Term Memory)**. This layer works very similarily to the simpleRNN layer but adds a way to access inputs from any timestep in the past. Another component that keeps track of the internal state. Until now the only thing we were tracking as kind of our internal state was the inmediatly previous output.

Whereas in our simple RNN layer input from previous timestamps gradually disappeared as we got further through the input. With a LSTM we have a **long-term memory data structure storing all the previously seen inputs** as well as when we saw them. 

Rather than just kipping track of the previous output, we will add all of the output that we have seen so far into what we are goint to call a kind of "**conveyor belt**". It is almost just as a lookup table that can tell us the output at any previous cell that we want.

**This allows for us to access any previous value (of the state) we want at any point in time.** This adds to the complexity of our network and allows it to discover more useful relationships between inputs and when they appear.
![](https://i.imgur.com/cEOnG0L.png)

---

## Sentiment Analysis
 
The formal definition of this term from Wikipedia is as follows:

The process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.

Tensorflow tutorial: https://www.tensorflow.org/tutorials/text/text_classification_rnn

---

## Sources

* Chollet François. Deep Learning with Python. Manning Publications Co., 2018.
* “Text Classification with an RNN  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/text/text_classification_rnn.
* “Text Generation with an RNN  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/text/text_generation.
* Understanding LSTM Networks.” Understanding LSTM Networks -- Colah's Blog, https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---
