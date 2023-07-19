<a name="readme-top"></a>


# IMDB Sentiment Analysis
## Project Description
Imagine a company you work for sells widely-used products, and customers rightfully leave reviews based on their experience with those products. Now imagine your job is to classify each review, and determine the pros and cons of each product to improve upon it. But now you have a very mind-numbing task ahead of you: to manually scan through thousands of reviews. Doesn't sound too fun right? This is where Sentiment Analysis is effective.

Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone or “sentiment” expressed in a piece of text. This can apply to a multitude of industries to analyze customer feedback, monitor brands, understand market trends, and overall allow businesses to extract valuable insights from vast amounts of textual data in a way that is much more efficient than humans parsing through millions of data. 

In this project, I created a Recurrent Neural Network (RNN) that takes in movie reviews from the popular platform IMDB and classifies the review as either “positive” or “negative” based on the sentiment analysis of the text. The reason why I used an RNN is because RNNs are designed to handle sequential data. Since words in sentences are related to one another and are not distinct entities, RNNS are capable of retaining information from previous time steps during the training process, making them suitable for capturing the relationships between words in a sentence. 

## Table of Contents
<!-- TABLE OF CONTENTS -->
<ol>
  <li>
    <a href="#data-preprocessing">Data Preprocessing</a>
    <ul>
      <li><a href="#cleaning-data">Cleaning data</a></li>
      <li><a href="#data-transformation">Data transformation</a></li>
      <li><a href="#encoding-the-output">Encoding the Output</a></li>
      <li><a href="#data-splitting">Data splitting</a></li>
    </ul>
  </li>
  <li>
    <a href="#building-the-model">Building the Model</a>
    <ul>
      <li><a href="#embedding-layer">Embedding layer</a></li>
      <li><a href="#bidirectional-lstm-layer">Bidirectional LSTM layer</a></li>
      <li><a href="#fully-connected-layer">Fully connected layer</a></li>
    </ul>
  </li>
  <li>
    <a href="#model-evaluation">Model Evaluation</a>
    <ul>
      <li><a href="#confusion-matrix">Confusion matrix</a></li>
    </ul>
  </li>
  <li><a href="#conclusion-and-future-directions">Conclusion and future directions</a></li>
</ol>


## Data Preprocessing

This project uses this Kaggle dataset of 50,000 movie reviews from IMDB. The dataset is a `.csv` file where the first column is the review text and the second column is whether it is "positive" or "negative.

Unfortunately, we cannot just use the raw dataset and feed it into our model. The data must be cleaned and transformed in a way suitable for analysis. 


### Cleaning Data

The first step is to "clean" the data and extract only the most important variables from the text. I performed these common text preprocessing steps:
<ul>
  <li>Removing punctuation</li>
  <li>Lowercase text</li>
  <li>Removing stop words
    <ul>
      <li> Stop words are the most common words in the English language such as “I”, “have”, “are”, etc. Alone, these words do not hold any significant meaning, and can thus skew our sentiment analysis. So, we remove all instances of these stop words.
      </li>
    </ul>
  </li>
  <li>Word stemming
    <ul>
      <li>Stemming reduces words to their root form by removing suffixes; so “walking” and “walked” becomes “walk”. This ensures that multiple versions of the same word are treated the same.
      </li>  
    </ul>
  </li>
</ul>
Additionally, this specific dataset has multiple instances of “<br>”, which were removed as well. 

### Data Transformation

Now that we have cleaned the data, we must transform the raw text into a form that deep learning models can understand. We do this through “Tokenization”, which breaks down text into individual words or “tokens”. These tokens are then converted to sequences of integers.

We do this using `Keras` tokenizer. In this particular instance of tokenization, `max_words = 5000` means that only the most frequent 5000 words will be kept, and less frequent words are discarded, and `max_len = 200` means that the maximum length of the sentences is 200 words. If a sentence is longer than 200, it will be truncated, and if it's shorter, it will be padded to reach the desired length.

### Encoding the output

An important preprocessing step is label encoding the output, which means representing categorical output variables with numerical values. In this context, “positive” and “negative” sentiments will be represented as “1” and “0” respectively. 

We do this using the LabelEncoder class from `Scikit learn (Sklearn)`, which automatically separates each output to its respective numerical representation.

### Data splitting

Finally, now that the input data is fully preprocessed and the output is encoded, we can split the dataset into its training and testing subsets using the train_test_split class from `Sklearn`.

## Building the Model

This RNN model is relatively simple with only 3 layers; however, each layer is vital to the RNN and serves a specific purpose.  
<ol>
  <li>Embedding layer
    <ul>
        <li> In the embedding layer, we convert words into dense vectors that pack information in few dimensions. Word embeddings represent words in a continuous vector space, where similar words are located closer to each other, meaning that words with similar meanings will have similar vector representations, which helps the model understand the context and identify sentiment-related words more effectively. The max_words parameter represents the size of the vocabulary, and each word is represented as a vector of length 40 (embedding dimension). The input_length parameter sets the length of the input sequences to be fed into the model, which should be max_len.
        </li>
    </ul>
  </li>
  <li>Bidirectional LSTM Layer
    <ul>
      <li> In the Bidirectional LSTM layer, we allow the LSTM layer to process the input sequence in both forward and backward directions, effectively capturing dependencies in both directions. The LSTM layer has 20 units (or cells), meaning it has 20 memory cells to remember information over time. The dropout parameter is set to 0.5, indicating that a dropout layer will be applied after the LSTM layer, which helps prevent overfitting by randomly setting a fraction of input units to zero during training.
      </li>
    </ul>
  </li>
  <li>Fully connected layer
    <ul>
      <li>The Dense layer is a fully connected layer, where each neuron is connected to every output from the previous layer. It has a single neuron because the task is binary classification, and a sigmoid activation function is used to produce an output between 0 and 1, representing the probability of the input belonging to the positive class (since the activation is "sigmoid"). We use the sigmoid activation function because the output is binary.
      </li>  
    </ul>
  </li>
</ol>
When fitting the model, I utilized a batch size of 128 for 10 Epochs. I also monitored the model's performance on a validation set during training. The model's performance on the validation data is used to monitor how well the model generalizes to new, unseen data and helps in preventing overfitting. If the validation loss starts to increase or stops improving, I utilize “early stopping” to stop the training early which further prevents overfitting.

## Model Evaluation

The model reached a final test accuracy of 0.8826, which is slightly lower than the highest training accuracy of 0.9118. Due to the early stopping, the model halted training after Epoch 4. 

### Confusion matrix
I computed the confusion matrix based on the predictions made by my model on the test dataset. The confusion matrix showed the model correctly classified 5513 positive reviews and 5520 negative reviews, yielding accuracies of `87.63%` and `88.90%` accuracy respectively. 

## Conclusion and future directions


With the model being incredibly simple, I hope to create a more complex RNN to achieve at least 95% test accuracy, as well as fine-tuning hyper parameters. Additionally, I am curious to see how effectively my model can categorize reviews into increasing categories, such as including a “neutral” output or even classifying reviews into their respective 1-5 star rating. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
