######### Switched from linear to logistic because3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import data_generator

synthetic_data = data_generator.data

# TODO : We dont want to evaluate the accuracy on the data the model is being trained!
# Rather we want to take a cut out of the data


# The feature columns
X = synthetic_data[["height", "fitness", "photo_quality", "bio_length", "smiling"]]

# The label
y = synthetic_data["match"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Checking the accuracy score ::",accuracy_score(y_test, predictions))

probabilities = model.predict_proba(X)[:, 1]

print(synthetic_data.head())
print(predictions[:5])
print(probabilities[:5])