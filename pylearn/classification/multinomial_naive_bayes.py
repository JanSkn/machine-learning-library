import numpy as np
import pandas as pd
import time
import re

# TODO count word in one sentence once or count occurances
class MultinomialNaiveBayes:
    """
    Computes text classification problems by applying the Bayes theorem.

    Parameters:
        :TODO
    """
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
        self.prior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=range(1))
        tokenized_texts = self._tokenize(self.texts)  # list of lists of words of one row
        self.vocab = self._create_vocab(tokenized_texts)    # contains all words
        self.posterior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=self.vocab)
        for index, c in enumerate(self.classes):
            class_df = X[Y == c]
            print(class_df)
            self.prior.iloc[index] = float(len(class_df)) / float(self.num_of_samples)
            tokenized_class_texts = self._tokenize(class_df)
            print(tokenized_class_texts)
            self.posterior.iloc[index] = self._bow(tokenized_class_texts)         
            print("before")
            print(self.posterior)
            # Laplace smoothing:
            # P(x|y) = (N_x,y + α) / (N_y + α * D)
            # α: smoothing parameter
            N_xy = self.posterior.iloc[index]           # N_x,y: number of x occuring in class y               
            N_y = len(tokenized_class_texts[0])         # N_y: number of features in y
            D = self.posterior.shape[1]                 # D: number of features in whole data set
            self.posterior.iloc[index] = (N_xy + alpha) / (N_y + alpha * D)
            print("after")
            print(self.posterior)

        end = time.time()

        if log_duration:
            print(f"Duration of training: {end - start}\n")
    
    def predict(self, X: pd.Series) -> pd.DataFrame:
        """
        Computes the output of a given X.

        Parameters:
            :X (pandas.Series): Testing input

        Returns:
            Predicted classes as pandas dataframe
        """
        y_pred = [self._predict(x[1:len(x)]) for x in X.items()]           # x is item object --> x[1:len(x)] removes index (iteritems deprecated)
        return pd.DataFrame(y_pred)                                         # TODO change to Series for easier usage/transformation to array
    
    def _predict(self, x: tuple) -> int | str:
        #print(x)
        x = self._tokenize(pd.Series(x))[0]             # get a list of all words
        #print(x)
        # P(y) * ∏ P(x_i|y) --> ln P(y) + ∑ ln P(x_i|y) to prevent underflow
        posterior = self.posterior[x].sum(axis=1) 
        # Laplace-Smoothing to avoid log of 0, wrong yet

        
        #if posterior
        for index, c in enumerate(self.classes):
            prior = np.log(self.prior.iloc[index])       # get prior at index
            posterior.iloc[index] = np.log(posterior.iloc[index]) + prior                
        print(posterior)

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
        print(bow_df.shape)
        print("tt", tokenized_texts)
        for i, text in enumerate(tokenized_texts):
            for word in text:
                bow_df.at[i, word] = 1
        # print("a")
        # print(bow_df)
        # for i, text in enumerate(tokenized_texts):
        #     bow_df.loc[i, text] = 1
        # print("b")
        # print(bow_df)
        # word_probability = {}

        # for column in bow_df.columns:
        #     for word in bow_df[column]:
        #         word_probability[column] = bow_df[column].sum() / self.num_of_samples     # TODO change to self.num_of_samples
        print("bow df")
        print(bow_df)
        word_probability = bow_df.sum(axis=0) #/ self.num_of_samples
   
        #print(bow_df.head())
        return pd.DataFrame([word_probability])    
        
    @staticmethod
    def _create_vocab(tokenized_texts: list) -> list:
        """
        Helper function for fit.
        """
        return list(set(word for text in tokenized_texts for word in text))
    
    @staticmethod
    def _tokenize(texts: pd.Series) -> list:
        """
        Helper function for fit.
        """
        def clean_text(text):
                # remove everything but letters and whitespace (note: falsifies some words, e.g. Gerard's --> Gerards)
                text = re.sub(r"[^a-zA-Z\s]", "", text.lower())     
                tokens = text.split()   # split words by whitespace
                return tokens

        texts = texts.tolist()
        return [clean_text(text) for text in texts]
        
    

if __name__ == "__main__":  
    data = pd.read_csv("examples/data/fake_news.csv")
    data = data[["title", "real"]]      # remove unneccessary columns
    
    nb = MultinomialNaiveBayes()
    nb.fit(None, None)
    # nb.fit(data["title"], data["real"])
    # print(nb.predict(data["title"]))
    