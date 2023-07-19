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


<!-- ABOUT THE PROJECT -->
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
      <li>Removing stop words</li>
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


<!-- GETTING STARTED -->
## Libraries

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

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
