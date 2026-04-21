######### Switched from linear to logistic because3

from sklearn.linear_model import LogisticRegression
from data import data_generator

synthetic_data = data_generator.data

# The feature columns
X = synthetic_data[["height", "fitness", "photo_quality", "bio_length", "smiling"]]

# The label
y = synthetic_data["match"]

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

print(synthetic_data.head())
print(predictions[:5])
print(probabilities[:5])