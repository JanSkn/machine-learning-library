import numpy as np
import pandas as pd
import time
import re

class MultinomialNaiveBayes:
    """
    Computes text classification problems by applying the Bayes theorem.

    Attributes:
        :texts (pandas.Series): Input texts used for training
        :classes (list): List of unique classes in the training data
        :num_of_samples (int): Total number of samples in the training data
        :prior (pandas.DataFrame): Prior probabilities of each class
        :vocab (list): List of unique words in the training data
        :posterior (pandas.DataFrame): Posterior probabilities of each word given each class
    """
    # loc selects rows by index label (can also be numeric, but numeric index can differ from real index), 
    # iloc selects row by actual index
    def fit(self, X: pd.Series, Y: pd.Series, alpha=1, log_duration=True) -> None:
        """
        Trains the algorithm. 
        Data must not be continuous data.

        Parameters:
            :X (pandas.Series): Training input
            :Y (pandas.Series): Training output
            :alpha (float, optional): Smoothing parameter
            :log_duration (bool, optional): Logs the duration of the training, default: True

        Returns:
            None
        """
        start = time.time()

        self.texts = X
        self.classes = sorted(list(Y.unique()))
        self.num_of_samples = len(X) 
        self.prior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=range(1)).astype(float)
        tokenized_texts = self._tokenize(self.texts)        # list of lists of words of one text
        self.vocab = self._create_vocab(tokenized_texts)    # contains all words
        self.posterior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=self.vocab).astype(float)
        for index, c in enumerate(self.classes):
            class_df = X[Y == c]
            self.prior.iloc[index] = len(class_df) / self.num_of_samples                # ratio of classes in the sample
            tokenized_class_texts = self._tokenize(class_df)                            # tokenized texts per class
            self.posterior.iloc[index] = self._bow(tokenized_class_texts)         
            # Laplace smoothing:
            # P(x|y) = (N_x,y + α) / (N_y + α * D)  
            # α: smoothing parameter
            N_xy = self.posterior.iloc[index]           # N_x,y: occuramce of feature (word) x in class y    
            N_y = 0                                     # N_y: number of features (words) in y
            for text in tokenized_class_texts:          
                N_y += len(text)
            D = self.posterior.shape[1]                 # D: number of features (words) in whole data set
            self.posterior.iloc[index] = (N_xy + alpha) / (N_y + alpha * D)
            
        end = time.time()

        if log_duration:
            print(f"Duration of training: {end - start}\n")
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Computes the output of a given X.

        Parameters:
            :X (pandas.Series): Testing input

        Returns:
            Predicted classes as array
        """
        y_pred = [self._predict(x[1:len(x)]) for x in X.items()]           # x is item object --> x[1:len(x)] removes index (iteritems deprecated)
        return np.array(y_pred)                                        

    def _predict(self, x: tuple) -> int | str:
        """
        Helper function for predict.

        Parameters:
            :x (tuple): A tuple with the tokenized text

        Returns:
            The class with the highest probability
        """
        # P(y) * ∏ P(x_i|y) --> ln P(y) + ∑ ln P(x_i|y) to prevent underflow

        x = self._tokenize(pd.Series(x))[0]             # get a list of all words
        prior = np.log(self.prior)                      # ln P(y)
        posterior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=x).astype(float)
        try:
            posterior = np.log(self.posterior[x])       # ln P(x_i|y)
        except KeyError:                                # self.posterior[x] raises KeyError if word not found, happens if vocab not big enough
            for word in x:
                if word not in self.posterior.columns:
                    posterior[word] = 1e-10             # add new column (word) with probability near 0
                else:
                    posterior[word] = self.posterior[word]
        posterior = posterior.sum(axis=1)               # ∑ ln P(x_i|y)
        posterior = prior[0] + posterior                # ln P(y) + ∑ ln P(x_i|y)       (prior[0] to change DataFrame to Series)
        return self.classes[np.argmax(posterior)]
    
    def _bow(self, tokenized_texts: list) -> pd.DataFrame:
        """
        Helper function for fit.
        Creates bag-of-words model from the vocabulary of a class.

        Parameters:
            :tokenized_texts (list): A list of the words of each sample

        Returns:
            The probabilities of each word as pandas dataframe
        """
        bow_df = pd.DataFrame(np.zeros((len(tokenized_texts), len(self.vocab))), columns=self.vocab)

        for i, text in enumerate(tokenized_texts):
            for word in text:
                bow_df.at[i, word] += 1

        word_probability = bow_df.sum(axis=0) 
   
        return pd.DataFrame([word_probability])    
        
    @staticmethod
    def _create_vocab(tokenized_texts: list) -> list:
        """
        Helper function for fit.

        Parameters:
            :tokenized_texts (list): A list of the words of each sample

        Returns:
            The vocabulary with all unique words
        """
        return list(set(word for text in tokenized_texts for word in text))
    
    @staticmethod
    def _tokenize(texts: pd.Series) -> list:
        """
        Helper function for fit.

        Parameters:
            :texts (pandas.Series): All texts in a series

        Returns:
            The texts as a list of lists of tokenized texts, input is cleaned from special characters
        """
        def clean_text(text):
                # remove everything but letters and whitespace (note: falsifies some words, e.g. Gerard's --> Gerards)
                text = re.sub(r"[^a-zA-Z\s]", "", text.lower())     
                tokens = text.split()   # split words by whitespace
                return tokens

        texts = texts.tolist()
        return [clean_text(text) for text in texts]