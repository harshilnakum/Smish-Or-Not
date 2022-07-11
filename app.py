import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Smishing/Spam Classifier")

input_sms = st.text_area("Enter the message: ")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.subheader("Spam!")
    else:
        st.subheader("Not Spam!")

else:
    st.text(" \n")
    st.text(" \n")
    st.text(" \n")

st.header("Rationale")
st.write("Texting is the most common use of smartphones. Experian found that adult mobile users aged 18 to 24 send more than 2,022 texts per month—on average, that's 67 per day—and receive 1,831. A couple of other factors make this a particularly insidious security threat.")
st.write('Smishing is a portmanteau of "SMS" (short message services, better known as texting) and "phishing." When cybercriminals "phish," they send fraudulent emails that seek to trick the recipient into opening a malware-laden attachment or clicking on a malicious link. Smishing simply uses text messages instead of email.')
st.write("Owing to these reasons it is of utmost importance to build an end-to-end system that can recognize and flag fraudulent spam messages. ")


st.header("Dataset")
st.write("We will be using the 'SMS Spam Collection Dataset' to feed our model and predict outcomes. The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam. The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.")
st.text("Code to transform csv to dataframes using pandas: ")
code = '''df = pd.read_csv('spam.csv',encoding='latin1')'''

st.code(code, language='python')

st.header("Data Cleaning")
st.write("Upon analysing the dataframe, we observe that there are three columns named 'Unnamed: 2', 'Unnamed: 3' and 'Unnamed: 4' which are largely empty and do not provide anything useful for our model. So we will proceed to remove these columns. ")
st.text("Code to drop useless columns: ")
code = '''df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)'''

st.code(code, language='python')

st.write("We will rename the columns v1 to target and v2 to text for easier understanding.")
code = '''df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)'''

st.code(code, language='python')

st.write("Next we will label the 'ham' as 0 and 'spam' as 1 using sklearn's preprocessing library.")
code = '''from sklearn.preprocessing import LabelEncoderencoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])'''

st.code(code, language='python')

st.write("Now we need to check for null and duplicate values and remove them. Upon analysing the dataframe we found 403 duplicates, fortunately there are no null values present. We will proceed to drop the duplicates.")
code = '''df = df.drop_duplicates(keep='first')'''

st.code(code, language='python')

st.header("Exploratory Data Analysis")

st.write("The main purpose of EDA is to help look at data before making any assumptions. It can help identify obvious errors, as well as better understand patterns within the data, detect outliers or anomalous events, find interesting relations among the variables.")
st.write("First we will see the distribution of spam and ham messages in the data")
code = '''import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()'''


st.code(code, language='python')
st.text("Output:")
image = Image.open('Screenshot_1.png')

st.image(image)

st.write("Here we notice that the data is very imbalanced as majority is ham and only 12% is spam. So we will proceed with a high precision model.")

st.write("Now we will add three new features into the data frame namely num_characters, num_words, num_sentences and try to analyse their relationship with the target label i.e. spam or ham.")
st.text("Code to add three new features using nltk library")
code = '''import nltk
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] =df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
'''


st.code(code, language='python')

st.write("We will proceed to plot some figures using seaborn library and try to establish a relationship between these new added features and target column.")
st.text("Code to plot num_characters new feature with respect to target values.")
code = '''import seaborn as sns
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_characters'], color='green')
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')
'''
st.code(code, language='python')
st.text("Output: ")
image = Image.open('Screenshot_3.png')

st.image(image)

st.text("Code to plot num_words new feature with respect to target values.")
code = '''import seaborn as sns
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_words'], color='green')
sns.histplot(df[df['target'] == 1]['num_words'], color='red')
'''
st.code(code, language='python')
st.text("Output: ")
image = Image.open('Screenshot_2.png')

st.image(image)

st.write("Using the above plots we observe that spam messages are generally of higher length character or words than a ham messages.")

st.write("Now we will plot a pairplot of all these three features with each other and try to churn out a hidden relationship.")
st.text("Code to pairplot the target as hue:")
code = '''sns.pairplot(df, hue='target')'''
st.code(code, language='python')
st.text("Output: ")
image = Image.open('Screenshot_4.png')

st.image(image)
st.write("We observe that most of the data is linearly distributed but there are some outliers present.")

st.write("Now we will plot a heatmap of these features.")
st.text("Code to plot the heatmap:")
code = '''sns.heatmap(df.corr(), annot=True)'''
st.code(code, language='python')
st.text("Output: ")
image = Image.open('Screenshot_5.png')

st.image(image)
st.write("We observe that multicollinearity is present in the data, we establish that we should use num_characters as our main feature because it has the highest collinearity coefficient among all three.")

st.header("Data Preprocessing")

st.write("We will convert all the text data into lowercase, then we will tokenize it using the nltk library. We can remove any special characters and stop words from the data as they do not add any meaning to our model. Finally we will stem the data which will convert all words into their root form.")
st.text("Function to transform the text: ")
code = '''def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)'''
st.code(code, language='python')


st.text("Code to apply transform_text function to data frame:")
code = '''df['transformed_text'] = df['text'].apply(transform_text)'''
st.code(code, language='python')

st.header("Model Building")

st.write("We will be using tfid vectorizer for this model. It was selected over CountVectorizer after performing various tests.")

st.text("Code to build three different models for three algorithms GaussianNB, MultinomialNB and BernoulliNB:")
code = '''from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2 )
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
'''
st.code(code, language='python')

st.write("Now we will print and analyse performance of these three algorithms:")

image = Image.open('Screenshot_6.png')

st.image(image)

image = Image.open('Screenshot_7.png')

st.image(image)
image = Image.open('Screenshot_8.png')

st.image(image)
image = Image.open('Screenshot_9.png')

st.image(image)
image = Image.open('Screenshot_10.png')

st.image(image)

st.write("Upon analysing these plots, we find that it will be most fruitful if we go for Multinomial Naive Bayes because it offers 100% Precision and 97% Accuracy. ")

st.header("Improving the model")

st.subheader("Voting Classifier")
st.write("Let's try to improvise the accuracy while maintaining the precision at 100%.")

st.text("Code for Voting Classifier:")
code = '''svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
'''
st.code(code, language='python')

st.text("Output: \n"
"Accuracy 0.9825918762088974\n"
"Precision 0.9918032786885246")

st.write("As we can observe, voting classifier didn't improve the accuracy while maintaining the precision. So we will reject voting classifier.")

st.subheader("Stacking")

st.write("As voting classifier has equal say to all three algorithms, we will try to assign a weighted contribution using stacking. We will use Random Forest Classifier for this implementation.")
st.text("Code for Stacking:")
code = '''estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
'''
st.code(code, language='python')

st.text("Output: \n"
"Accuracy 0.9796905222437138\n"
"Precision 0.9465648854961832")

st.write("We observe that stacking too didn't improve the accuracy without degrading the precision. Hence, we will stick to our original model which had 100% precision and 97% accuracy.")