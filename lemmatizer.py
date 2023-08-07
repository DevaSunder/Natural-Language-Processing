import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



paragraph = """The Taj Mahal was built by Emperor Shah JahanBursting with imagery, motion, interaction and distraction though it is, today’s World Wide Web is still primarily a conduit for textual information. In HTML5, the focus on writing and authorship is more pronounced than ever. It’s evident in the very way that new elements such as article and aside are named. HTML5 asks us to treat the HTML document more as… well, a document.

It’s not just the specifications that are changing, either. Much has been made of permutations to Google’s algorithms, which are beginning to favor better written, more authoritative content (and making work for the growing content strategy industry). Google’s bots are now charged with asking questions like, “Was the article edited well, or does it appear sloppy or hastily produced?” and “Does this article provide a complete or comprehensive description of the topic?,” the sorts of questions one might expect to be posed by an earnest college professor.

This increased support for quality writing, allied with the book-like convenience and tactility of smartphones and tablets, means there has never been a better time for reading online. The remaining task is to make the writing itself a joy to read."""


#Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer() # For Stemming Purpose



for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]
    sentences[i] = " ".join(words)

print(sentences[0])