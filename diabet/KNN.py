import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# csv doysayısı import edildiği yer.
diabets = pd.read_csv('diabets4.csv', sep = ';')

# verilerimizin ilk birkaç satırını yazdırıyoruz.Değerler doğru gelmişmidir?
a = diabets.head()

b = diabets.info()

c = diabets.describe() #describe değerleri


##########################################################
#1 Pregnancies                = hamilelik                #
#2 Glucose                    = Glikoz                   #
#3 BloodPressure              = Kan basıncı              #
#4 SkinThickness              = Deri Değeri              #
#5 Insulin                    = İnsülin değeri           #
#6 BMI                        = Vicut Kitle Endeksi      #
#7 DiabetesPedigreeFunction   = diyabet soyağacı         #
#8 Age                        = Yaş                      #
#9 Outcome                    = Çıktı (pozitif, negatif) #
##########################################################

# 1: Öğrenmenin gerçekleneceği özelikler ve çıktı. arraylere atanıyor.
features = diabets[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
features = features.to_numpy() # özellik kümesini numpy dizisine dönüştürür
target = diabets['Outcome'].to_numpy() # hedef sütunu numpy dizisine dönüştürür

# 2: verisetinin özelliklerini standardize eden fonksiyon.
def standardScaler(feature_array):
    """numpy olarak gelen arrayleri standardasyon için ölçekler
     
    Args-
        verilerim özellik değerleri. (BMI, Insulin...) gibi değerler
        standardasyon için gönderilir.
    
    Returns-
        feature_array- standardasyon tamamlandıktan sonraki çıktı.
    """
    
    total_cols = feature_array.shape[1] # toplam sütun sayısı 
    for i in range(total_cols): # her situnda 1 iterasyon gerçekle
        feature_col = feature_array[:, i]
        mean = feature_col.mean() # ortalamayı bulur.
        std = feature_col.std() # standard sapma
        feature_array[:, i] = (feature_array[:, i] - mean) / std # ölçekleme
    return feature_array

# 3
features_scaled = standardScaler(features) 
features_scaled

sns.countplot('Outcome', data = diabets)

def train_test_split(features, target, test_size = 0.20):
    """Veri kümesini eğitim ve test kümelerine böler.
    
    Args- 
        features- Özellik dizisi
        target- hedef dizi
        test_size- test veri kümesi boyutu
    
    Returns-
        train_features, train_target, test_features, test_target 
    """
    num_total_rows = features.shape[0] # veri kümesindeki toplam satır sayısı
    num_test_rows = np.round(num_total_rows * test_size) # test veri kümesindeki toplam satır
    rand_row_num = np.random.randint(0, int(num_total_rows), int(num_test_rows)) # rastgele oluşturulmuş Satır Numaraları
    
    # train and test
    test_features = np.array([features[i] for i in rand_row_num]) # test kümesi
    train_features = np.delete(features, rand_row_num, axis = 0) # test veri satırlarını Ana Veri kümesinden siler. eğitim veri kümesi yapar

    # train and test target
    test_target = np.array([target[i] for i in rand_row_num]) # train kümesi
    train_target = np.delete(target, rand_row_num, axis = 0) # training target listesi
    
    return train_features, train_target, test_features, test_target 

# dataset için verilerin train ve test olarak bölünmesini çağırdık.
X_train, y_train, X_test, y_test = train_test_split(features_scaled, target, test_size = 0.2)

# split edilmiş veriler.
shape = X_train.shape, y_train.shape, X_test.shape, y_test.shape



def euclidean_dist(pointA, pointB):
    """iki nokta vektörü arasındaki öklit uzaklığı.
    Args-
        pointA- A vektörü
        pointB- B vektörü
    Returns-
        distance- A noktası ile B arasındaki öklit uzunluğu
    """
    distance = np.square(pointA - pointB) # (ai-bi)**2 
    distance = np.sum(distance)
    distance = np.sqrt(distance) 
    return distance

def distance_from_all_training(test_point):
    """datasetteki her 2 nokta arasındaki öklid hesaplanıyor. euclidean_dist fonk. çağırılır.
    Args- 
        test_point- test veri seti
    Returns- 
        dist_array- eğitim verileri için distance hesabı
    """
    dist_array = np.array([])
    for train_point in X_train:
        dist = euclidean_dist(test_point, train_point)
        dist_array = np.append(dist_array, dist)
    return dist_array

dist_array = distance_from_all_training(X_test[0])
dist_array


def KNNClassifier(train_features, train_target, test_features, k = 5):
    """Test kümasini KNN ile sınaflar.
    Args- 
        train_features- eğitim verilerinin özellik kümasi (inusilin, BMI gibi)
        train_target- eğitim kümasinin çıktısı (Outcome)
        test_features- test verilerinin özellik kümasi (inusilin, BMI gibi)
        k =5 - en yakın komşuluk k değeri kadar!!!
    Returns-
        predictions- her test verisi için tahmin dizisi 
    """
    predictions = np.array([])
    train_target = train_target.reshape(-1,1)
    for test_point in test_features: # test verilerini dön 
        dist_array = distance_from_all_training(test_point).reshape(-1,1) # train verilerinin mesafesi hesaplanıyor.
        neighbors = np.concatenate((dist_array, train_target), axis = 1) 
        neighbors_sorted = neighbors[neighbors[:, 0].argsort()] # Eğitim verilerini öklit mesafesine göre sıralar
        k_neighbors = neighbors_sorted[:k] # en yakın komşuları belirler.
        frequency = np.unique(k_neighbors[:, 1], return_counts=True)
        target_class = frequency[0][frequency[1].argmax()] # en yüksek değere sahip label seçilir.
        predictions = np.append(predictions, target_class)
    
    return predictions

# running inference on the test data
test_predictions = KNNClassifier(X_train, y_train, X_test, k = 5)
test_predictions

def accuracy(y_test, y_preds):
    """Modelin çıkarım doğruluğunu hesaplar.
    
    Args-
        y_test- test kümesinin orijinal hedef etiketleri
        y_preds- öngörülen hedef etiketleri
    Returns-
        acc
    """
    total_correct = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == int(y_preds[i]):
            total_correct += 1
    acc = total_correct/len(y_test)
    return acc

acc = accuracy(y_test, test_predictions)
print('Model accuracy (default k=5) = ', acc*100)

# sklearn ile doğruluğu kontrol etmek istersek aşağıyı aktif etmeliyiz.########
"""
from sklearn.neighbors import KNeighborsClassifier as KNN

model = KNN()
model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy(y_test, preds)
print('Model accuracy (Sklearn) = ', acc*100)
"""
###############################################################################

k_values = list(range(1,20))
accuracy_list = []
for k in k_values:
    test_predictions = KNNClassifier(X_train, y_train, X_test, k)
    accuracy_list.append(accuracy(y_test, test_predictions))
    
# plotting k-value vs accuracy plot
sns.barplot(k_values, accuracy_list)

# running inference for k=8
test_predictions = KNNClassifier(X_train, y_train, X_test, k = 8)

# checking the accuracy
acc = accuracy(y_test, test_predictions)
print('Model accuracy (k = 8) = ', acc*100)




import pickle
filename = 'diabets4.sav'

pickle.dump(test_predictions, open(filename, 'wb'))

#Modelin çağırılması
loaded_model = pickle.load(open(filename, 'rb'))

