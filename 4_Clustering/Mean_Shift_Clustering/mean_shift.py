import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# Importing dataset
df = pd.read_csv("Dataset/top50.csv", encoding='ISO-8859-1')

# Independent variable
X = pd.DataFrame(df.iloc[:, 4:13])

# Feature Scaling
normalised = StandardScaler()
X = normalised.fit_transform(X)

# Applying MeanShift for Classification
ms = MeanShift(0.5)
ms_result = ms.fit_predict(X)

# Visualizing clusters
X = pd.DataFrame(X)
plt.scatter(X.iloc[:, 7], X.iloc[:, 5], c=ms_result)
plt.show()
