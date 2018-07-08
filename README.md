# languageDetector
This is a multi-class random forest based language detector that works by using words, bigrams and trigrams as features. Support is provided for 7 languages using the Latin Script (English, German, Portuguese, Spanish, French, Dutch, Italian). 
To optimize training and testing performance by reducing number of features,a corpus of 300,000 sentences for each language is leveraged from [Leipzig Corpus] (http://wortschatz.uni-leipzig.de/en/download) and the 50 most frequent words, bigrams, and trigrams are shortlisted as features. The dataframe creation is slightly complicated, but it is highly vectorized to speed up performance. All train and test datapoints are then represented in the reduced feature-space. A model trained on 5,000 sentences from each language takes less than 2 minutes to train, and performs at 98% accuracy. To replicate the environment, please place the following data files sourced from this [Google Drive directory] (https://drive.google.com/drive/folders/1x2_8sInW4vMpkI0cgN-CALYwxR_XUCdh), and assign that to 'dirname'
- deu_mixed-typical_2011_300K-sentences.txt
- eng_news_2005_300K-sentences.txt
- fra_mixed_2009_300K-sentences.txt
- ita_mixed-typical_2017_300K-sentences.txt
- nld_mixed_2012_300K-sentences.txt
- por_newscrawl_2011_300K-sentences.txt
- spa_news_2006_300K-sentences.txt

### Performance
The longest task is that of finding most common features for every language (~ 1 minute per language). The training dataframe creation then takes ~ 2 minutes, and creating the random forest model takes ~ 1 minute. The following performance metrics are calculated 

- 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑟𝑢𝑒 𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠/(𝑇𝑟𝑢𝑒 𝑃𝑜𝑠𝑡𝑖𝑣𝑒𝑠+𝐹𝑎𝑙𝑠𝑒 𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠) 
- 𝑅𝑒𝑐𝑎𝑙𝑙= 𝑇𝑟𝑢𝑒 𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠/(𝑇𝑟𝑢𝑒 𝑃𝑜𝑠𝑡𝑖𝑣𝑒𝑠+𝐹𝑎𝑙𝑠𝑒 𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒𝑠)
- 𝐹1 𝑆𝑐𝑜𝑟𝑒= 2∗ 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛∗ 𝑅𝑒𝑐𝑎𝑙𝑙𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛+𝑅𝑒𝑐𝑎𝑙𝑙

In general, the model performs with 98% Precision, 98% Recall and 98% F1-Score. 

![alt text](https://github.com/kiranramnath007/languageDetector/blob/master/Classification%20report.PNG)

### Confusion matrix
The performance is also examined using the confusion matrix, that tells us the distribution of predicted labels v/s actual labels.

![alt text](https://github.com/kiranramnath007/languageDetector/blob/master/Confusion%20Matrix.PNG)

### Novel ideas
Using all bigrams, trigrams, and words will blow up the feature space and impact performance adversely. Hence the features are first shortlisted on the basis of most frequent features. This results in optimal performance both in terms of model accuracy and time taken.

### Scope for improvement

There is a need to prune feature space to further remove redundancies. One approach could be through the use of maximal substrings. For eg - the trigram ' a ' will be a substring of ' a' always and can be removed. The size of the training data (for feature shortlisting) can be reduced and an optimal size of data can be explored. Furthermore, the accuracy of the model for French and Portuguese can be bettered through use of slightly more features for these two languages in particular.
