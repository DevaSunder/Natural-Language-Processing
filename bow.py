import nltk




paragraph = """The Taj Mahal was built by Emperor Shah JahanBursting with imagery, motion, interaction and distraction though it is, today’s World Wide Web is still primarily a conduit for textual information. In HTML5, the focus on writing and authorship is more pronounced than ever. It’s evident in the very way that new elements such as article and aside are named. HTML5 asks us to treat the HTML document more as… well, a document.

It’s not just the specifications that are changing, either. Much has been made of permutations to Google’s algorithms, which are beginning to favor better written, more authoritative content (and making work for the growing content strategy industry). Google’s bots are now charged with asking questions like, “Was the article edited well, or does it appear sloppy or hastily produced?” and “Does this article provide a complete or comprehensive description of the topic?,” the sorts of questions one might expect to be posed by an earnest college professor.

This increased support for quality writing, allied with the book-like convenience and tactility of smartphones and tablets, means there has never been a better time for reading online. The remaining task is to make the writing itself a joy to read."""

#cleaning the texts
import re # Regular Expression
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

#Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
final=[]
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i]) # re.sub used to replace all the characters other than a-zA-Z with " ".
    review = review.lower()
    review = review.split() #We get a list of words
    review =[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    final.append(review)

#print("Original sentences \n", sentences)

#print("Processed sentences \n", final)

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(final).toarray()
print(X) #The vector format - Document Matrix