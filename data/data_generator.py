### We need to generate data LR

import pandas as pd ### Pandas is used to handle the data
import numpy as np ### numpy is used to deal with numbers


# Why are we seeding with 42 and no other number
np.random.seed(42)
n = 100

data = pd.DataFrame({
    "height" : np.random.normal(5.8, 0.3, n),
    "fitness" : np.random.randint(1, 10, n),
    "photo_quality" : np.random.randint(1, 10, n),
    "bio_length" : np.random.randint(20, 200, n),
    "smiling" : np.random.randint(0, 2, n),
})



# This is a derived attribute
# We give different important to different fields
# And generate a score for each profile
data ["match_score"] = (
    0.3 * data["photo_quality"] + # TODO : WHy different numbers
    0.2 * data["photo_quality"] +
    0.1 * data["photo_quality"] +
    np.random.normal(0, 1, n) ###### TODO : why this????
)




'''
TODO :
    The match score is used to generate the label and that's it
    The model is not aware about it and it is just for our convince 
    It will be a wrong thing to do in real but we will have the label in real!
'''

# This is called normalisation & we brought the values in range 0-1
# Easier to interpret
min_val = data["match_score"].min()
max_val = data["match_score"].max()
data["match_score"] = (data["match_score"] - min_val) / (max_val - min_val)

data["match"] = (data["match_score"] >= 0.5).astype(int)


print(data)





