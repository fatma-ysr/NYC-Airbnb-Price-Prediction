import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import time
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV

# ==========================================
#  VERİNİN HİKAYESİ VE DEĞİŞKEN SEÇİMİ
# ==========================================
# Bu projede, New York City (NYC) Airbnb açık veri seti kullanılarak konut bilgilerine 
# dayalı veri temizleme, özellik mühendisliği (feature engineering) ve fiyat tahminleme 
# süreçleri yürütülmüştür.Hedef değişkenimiz 'price' (fiyat) olup, analize konum, 
# oda tipi, popülerlik ve başlık metinleri gibi özellikler dahil edilmiştir.

# Kaggle giriş dizinindeki dosyaları listeliyoruz
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ==========================================
#  VERİ YÜKLEME VE DEĞİŞKEN SEÇİMİ
# ==========================================
# Veri seti Kaggle'ın standart giriş yolundan okunuyor.
# Dosya yolunu otomatik kontrol et
kaggle_path = "/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"
local_path = "C:/Users/Source Tech Co/Downloads/AB_NYC_2019.csv"

if os.path.exists(kaggle_path):
    df_raw = pd.read_csv(kaggle_path)
    print("✅ Veri Kaggle sunucusundan yüklendi.")
elif os.path.exists(local_path):
    df_raw = pd.read_csv(local_path)
    print("✅ Veri yerel bilgisayardan yüklendi.")
else:
    print("❌ HATA: Veri dosyası belirtilen yolların hiçbirinde bulunamadı!")

# df.head() fonksiyonu, veri setinin ilk 5 satırını döndürür.
# KULLANIM AMACI: Veri setinin genel yapısını anlamak ve hızlı bir "Eksik Değer" 
# (Missing Value) ön kontrolü yapmaktır. Örneğin; 'last_review' ve 'reviews_per_month' 
# sütunlarındaki NaN (eksik veri) değerleri bu ilk bakışta kolayca tespit edilebilir.
print(df_raw.head())

# --- ÇIKARILAN DEĞİŞKENLER (DROPPED FEATURES) VE GEREKÇELERİ ---

# 'id' ve 'host_id': 
# Bu sütunlar veri setindeki satırları ve kişileri tanımlayan benzersiz (unique) 
# sayılardır. İstatistiksel olarak hiçbir varyans veya tahmin gücü taşımazlar. 
# Modelin bunları "anlamlı bir sayı" zannedip aşırı öğrenme (overfitting) yapmasını 
# engellemek için çıkarılmıştır.

# 'host_name': 
# Ev sahibinin ismi fiyata doğrudan etki eden nicel bir değer değildir. 
# Ayrıca çok yüksek kardinaliteye (binlerce farklı isim) sahip olduğu için 
# modelde gürültü (noise) yaratacaktır.

# 'last_review': 
# Tarih formatındaki bu veri çok fazla eksik değere sahiptir. Tarihsel etkisi, 
# zaten elimizde olan 'reviews_per_month' değişkeni tarafından dolaylı olarak 
# temsil edildiği için model karmaşıklığını azaltmak adına elenmiştir.

# ---  SEÇİLEN / KRİTİK DEĞİŞKENLER (SELECTED FEATURES) VE GEREKÇELERİ ---

# 'neighbourhood_group' (Bölge) & 'latitude/longitude' (Enlem/Boylam): 
# Gayrimenkul piyasasının en temel belirleyicisi konumdur. Bu üç değişken, 
# hem kategorik hem de koordinat bazlı olarak lokasyon etkisini modele aktarır.

#  'room_type' (Oda Türü): 
# Evin tamamı mı yoksa sadece bir odası mı kiralandığı, fiyat üzerindeki en keskin 
# ayrıştırıcı faktördür.

# 'minimum_nights' & 'availability_365': 
# Bu değişkenler evin arz ve talep dengesini temsil eder. Uzun süreli konaklama 
# zorunluluğu veya evin yıl boyu müsait olup olmaması, fiyatlandırma stratejisinin 
# bir parçasıdır.

#  'number_of_reviews' & 'reviews_per_month': 
# İlanın popülerliğini ve güvenilirliğini gösteren sosyal kanıt (social proof) 
# metrikleridir. Yüksek etkileşim genellikle fiyatı stabilize eder veya artırır.

#  'calculated_host_listings_count': 
# Ev sahibinin profesyonel bir işletme mi yoksa bireysel bir kullanıcı mı olduğunu 
# gösterir. Profesyonel hostlar genellikle daha optimize ve piyasa odaklı 
# fiyatlandırma yaparlar.

columns_to_keep = [
    "name", "price", "neighbourhood_group", "room_type", "minimum_nights", 
    "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", 
    "availability_365", "latitude", "longitude"
]
df = df_raw[columns_to_keep].copy()

# ==========================================
# GÖRSELLEŞTİRME VE YORUMLAMASI
# ==========================================
# Modeli kurmadan önce ham verideki ilişkileri ve dağılımları anlamak için ilk analiz (EDA).
print("Veri görselleştirme grafikleri hazırlanıyor")

plt.figure(figsize=(18, 5))

# Koordinatlara göre evlerin harita üzerindeki dağılımı
plt.subplot(1, 3, 1)
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', 
                data=df[df['price'] < 500], palette='Set1', s=10, alpha=0.6)
plt.title("Semtlere Göre Ev Dağılımı")

# Semtlerin fiyatlara etkisi
plt.subplot(1, 3, 2)
sns.boxplot(x='neighbourhood_group', y='price', data=df[df['price'] < 500], palette='Set2')
plt.title("Semtlerin Fiyat Kutu Grafiği (Fiyat < $500)")
plt.xticks(rotation=45)

# Çoklu doğrusal bağlantıyı kontrol etmek için korelasyon matrisi
plt.subplot(1, 3, 3)
sayisal_df = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
sns.heatmap(sayisal_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Sayısal Değişkenler Korelasyon Matrisi")

plt.tight_layout()
plt.show()

# --SEMTLERE GÖRE EV DAĞILIMI (COĞRAFİ DAĞILIM)--
# Veri setindeki 'latitude' ve 'longitude' koordinatlarının doğruluğunu teyit etmek amaciyla kullanilmistir.
# Scatter plot incelendiğinde, New York'un coğrafi yapısı net bir şekilde ortaya çıkmaktadır. 
# Manhattan (kırmızı) merkezde yoğun bir kümelenme sergilerken, Staten Island (mor) 
# daha seyrek ve ana karadan kopuk bir dağılım göstermektedir.

# --SEMTLERİN FİYAT KUTU GRAFİĞİ (BOXPLOT - Fiyat < $500)--
# Kategorik bir değişken olan 'Semt' ile sürekli bir değişken olan 'Fiyat' arasındaki 
# istatistiksel ilişkiyi anlamak, medyan farklarını görmek ve aykırı değerlerin (outliers) 
# yayılımını tespit etmek için yapılmıştır.
# Manhattan'ın medyan fiyat çizgisinin diğer tüm semtlerden yukarıda olması, bölgenin 
# piyasa değerini doğrular. Tüm semtlerde kutuların üzerindeki yoğun noktalar, 
# $500 altındaki segmentte dahi çok sayıda "lüks" aykırı değer olduğunu gösterir. 
# Dağılımın sağa çarpık olması, regresyon modellerinde neden logaritmik dönüşüm yapmamız gerektiğini soyler

# --SAYISAL DEĞİŞKENLER KORELASYON MATRİSİ (HEATMAP)--
# 'price' değişkeninin diğer sayısal özelliklerle düşük korelasyon (r < 0.10) 
# göstermesi, fiyatın doğrusal bir yapıdan ziyade karmaşık ve doğrusal olmayan 
# (non-linear) bir yapıda olduğunu ispatlar. Bu durum, basit doğrusal regresyon yerine 
# neden XGBoost veya LightGBM gibi ağaç tabanlı algoritmaların tercih edildiğinin 
# istatistiksel gerekçesidir. 'number_of_reviews' ile 'reviews_per_month' arasındaki 
# orta-güçlü (0.55) ilişki ise beklenen bir popülerlik göstergesidir.


# ==========================================
#  UÇ DEĞER TEMİZLİĞİ
# ==========================================
# Fiyatı sıfır veya negatif olan teknik hatalı veriler filtreleniyor
df = df[df['price'] > 0].copy()

# Isolation Forest ile istatistiksel çok boyutlu uç değer (anomali) tespiti
outlier_data = df[['price', 'latitude', 'longitude', 'minimum_nights']]
iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
outlier_preds = iso_forest.fit_predict(outlier_data)

# Sadece normal dağılıma uyan (1) veriler ayrılıyor
df = df[outlier_preds == 1].copy()

# ==========================================
# LOGARİTMİK DÖNÜŞÜM
# ==========================================
# Fiyat değişkenindeki sağa çarpıklığı (skewness) gidermek için logaritma alınıyor
df['log_price'] = np.log(df['price'])
df = df.drop(columns=['price']) 

# ==========================================
# METİN MADENCİLİĞİ
# ==========================================
# İlan başlıklarındaki gizli özellikleri çıkarmak için TF-IDF algoritması kullanılıyor
df['name'] = df['name'].fillna("") 

vectorizer = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
kelime_matrisi = vectorizer.fit_transform(df['name'])
kelimeler = vectorizer.get_feature_names_out()

# Ağırlıklandırılmış 100 kelime matrisi ana veriye ekleniyor
df_kelimeler = pd.DataFrame(kelime_matrisi.toarray(), columns=[f"word_{k}" for k in kelimeler], index=df.index)
df = pd.concat([df, df_kelimeler], axis=1)

df = df.drop(columns=['name'])

# ==========================================
# YENİ DEĞİŞKEN ÜRETİMİ
# ==========================================
# Haversine formülü ile doğrudan Manhattan merkezine olan kuş uçuşu mesafe (KM)

# Veri setinde yer alan 'latitude' (enlem) ve 'longitude' (boylam) koordinatları, 
# ağaç tabanlı algoritmalar için tek başlarına bütünsel bir coğrafi anlam ifade etmez. 
# Modelin konum avantajını matematiksel olarak kavrayabilmesi için bu koordinatların 
# sürekli (continuous) bir mesafe değişkenine dönüştürülmesi amaçlanmıştır.
 
# Referans Noktası (Manhattan Merkezi):
# Veri seti New York'un 5 bölgesini (Manhattan, Brooklyn, Queens, Bronx, Staten Island) 
# içeriyor olsa da, emlak ve turizm piyasasında fiyat varyansını belirleyen ana odak 
# noktası Manhattan'dır. Evin bulunduğu semtten bağımsız olarak, "merkeze yakınlık" 
# ilkesinin fiyata etkisini ölçebilmek adına Manhattan'ın merkezi 
# (Empire State Binası: 40.748817, -73.985428) sabit referans noktası olarak belirlenmiştir.

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df['distance_to_center_km'] = haversine_distance(df['latitude'], df['longitude'], 40.748817, -73.985428)

# Etkileşim Değişkeni: Oda Türü ve Semt Kombinasyonu
# Modellerin, ilanın oda türü ('room_type') ve bulunduğu semt ('neighbourhood_group') 
# arasındaki çapraz fiyat ilişkisini (örn. Manhattan'daki Özel Oda ile Bronx'taki Özel Oda farkı) 
# yakalaması için iki değişken birleştirilmiştir.
df['room_neighbourhood'] = df['neighbourhood_group'] + "_" + df['room_type']

# Veri Gruplama (Binning): Konaklama Süresi
# İlana ait minimum konaklama süresindeki ('minimum_nights') uç değerlerin gürültüsünü 
# azaltmak için, konaklama süreleri 'kısa', 'orta' ve 'uzun' dönem olarak kategorize edilmiştir.
def bin_nights(n):
    if n <= 3: return 'short_term'
    if n <= 14: return 'medium_term'
    return 'long_term'
df['stay_category'] = df['minimum_nights'].apply(bin_nights)

# Oransal Değişken (Ratio Feature): Popülerlik Skoru
# İlanın gerçek popülerliğini hesaplamak için, evin aldığı toplam yorum sayısı ('number_of_reviews'), 
# o evin yıl içindeki müsaitlik gününe ('availability_365') oranlanmıştır.
df['popularity_score'] = df['number_of_reviews'] / (df['availability_365'] + 1)

# İkili Değişken (Binary Flag): Profesyonel Ev Sahibi Ayrımı
# Ev sahibine ait platformdaki toplam ilan sayısına ('calculated_host_listings_count') bakılarak, 
# birden fazla evi olan ticari ev sahiplerini (1) amatör/bireysel ev sahiplerinden (0) 
# ayırt edecek bir bayrak (flag) eklenmiştir.
df['is_pro_host'] = df['calculated_host_listings_count'].apply(lambda x: 1 if x > 1 else 0)

# Oransal Değişken (Ratio Feature): Yorum Yoğunluğu
# İlanın güncel talep yoğunluğunu görmek için, ilanın aylık yorum alma hızı ('reviews_per_month'), 
# evin aldığı toplam yorum sayısına ('number_of_reviews') oranlanmıştır.
df['review_density'] = df['reviews_per_month'] / (df['number_of_reviews'] + 1)

# Eksik veri atamasında hata (Infinity) almamak için ön kontrol
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================
#  ÇARPIKLIK GRAFİĞİ VE YENİ DEĞİŞKENLERİN ANALİZİ
# ==========================================
# Logaritma dönüşümünün veriyi nasıl normalleştirdiğinin görsel kontrolü
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_raw[df_raw['price'] > 0]['price'], bins=50, color='salmon', kde=True)
plt.title("Orijinal Fiyat Dağılımı")

plt.subplot(1, 2, 2)
sns.histplot(df['log_price'], bins=50, color='lightgreen', kde=True)
plt.title("Log Dönüşümü Sonrası Dağılım")
plt.tight_layout()
plt.show()

# ==========================================
#  EKSİK VERİ ATAMA
# ==========================================
# Verilerdeki boşluklar MICE (IterativeImputer) algoritmasıyla istatistiksel tahminle dolduruluyor
kategorik_sutunlar = ['neighbourhood_group', 'room_type', 'room_neighbourhood', 'stay_category']

df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True)

imputer = IterativeImputer(random_state=42, max_iter=10)
df_imputed_array = imputer.fit_transform(df_encoded)
df_clean = pd.DataFrame(df_imputed_array, columns=df_encoded.columns)

# ==========================================
#  TEST SETİ AYRIMI
# ==========================================
# Modelin ezberlemesini önlemek için veri %80 Eğitim, %20 Test olarak ayrılıyor
X = df_clean.drop(columns=['log_price'])
y = df_clean['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)


# ==============================================================================
# MODELLERİN UYGULANMASI VE OPTİMİZASYONU 
# ==============================================================================

# ---  XGBOOST VE RANDOMIZEDSEARCHCV ---
print("XGBoost modeli kuruluyor ve hiperparametre optimizasyonu yapma")

param_dist_xgb = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'subsample': [0.8, 1.0]
}

xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=123)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist_xgb,
    n_iter=15,          
    cv=3,               
    scoring='r2',       
    n_jobs=-1,          
    random_state=123,
    verbose=1           
)

# --- KRONOMETRE (XGB) ---
start_time_xgb = time.time()
random_search.fit(X_train, y_train)
best_xgb_model = random_search.best_estimator_
xgb_time = time.time() - start_time_xgb
print(f" XGBoost Optimizasyonu Tamamlandı! İşlem Süresi: {xgb_time:.1f} saniye")


# --- LIGHTGBM ---
print(" LightGBM eğitiliyor (HalvingRandomSearchCV ile)")
start_time_lgbm = time.time()

lgbm_param_dist = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [5, 7, 10, -1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.6, 0.8, 1.0]
}

lgbm_base = lgb.LGBMRegressor(random_state=123, n_jobs=-1, verbose=-1)

lgbm_search = HalvingRandomSearchCV(
    estimator=lgbm_base, 
    param_distributions=lgbm_param_dist, 
    factor=3, 
    min_resources=500,
    cv=3, 
    scoring='r2', 
    random_state=123, 
    n_jobs=-1
)
lgbm_search.fit(X_train, y_train)

lgbm_time = time.time() - start_time_lgbm
best_lgbm = lgbm_search.best_estimator_

# --- RANDOM FOREST ---
print(" Random Forest eğitiliyor (HalvingRandomSearchCV ile)")
start_time_rf = time.time()

rf_param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_base = RandomForestRegressor(random_state=123, n_jobs=-1)

rf_search = HalvingRandomSearchCV(
    estimator=rf_base, 
    param_distributions=rf_param_dist, 
    factor=3, 
    min_resources=500,
    cv=3, 
    scoring='r2', 
    random_state=123, 
    n_jobs=-1
)
rf_search.fit(X_train, y_train)

rf_time = time.time() - start_time_rf
best_rf = rf_search.best_estimator_

print(" Tüm modellerin optimizasyonu tamamlandı!")


# ==============================================================================
#  SONUÇLARIN YORUMLANMASI VE KARŞILAŞTIRMA
# ==============================================================================

print(" Modeller test ediliyor ve final metrikleri hesaplma")

# Karşılaştırılacak modellerin sözlüğü 
modeller = {
    "XGBoost": best_xgb_model,
    "LightGBM": best_lgbm,
    "Random Forest": best_rf
}

model_sonuclari = []
n_test = X_test.shape[0]
k_features = X_test.shape[1]
real_price = np.exp(y_test) # Gerçek dolar değerleri

# Her model için tek tek tahmin yapıp metrikleri hesaplayan döngü
for isim, model in modeller.items():
    # 1. Tahmin Yap
    log_preds = model.predict(X_test)
    
    # 2. Logaritmik fiyatı gerçek Dolara çevir (Anti-Log)
    preds_price = np.exp(log_preds)
    
    # 3. Hata ve Başarı Metriklerini Hesapla
    mae = mean_absolute_error(real_price, preds_price)
    rmse = np.sqrt(mean_squared_error(real_price, preds_price))
    r2 = r2_score(y_test, log_preds)
    adj_r2 = 1 - ((1 - r2) * (n_test - 1) / (n_test - k_features - 1))
    
    # 4. Optimizasyon sürelerini çek
    if isim == "LightGBM":
        sure = lgbm_time
    elif isim == "Random Forest":
        sure = rf_time
    elif isim == "XGBoost":
        sure = xgb_time
    
    # Sonuçları sözlük olarak listeye ekle (Tüm modeller artık aynı standartta kaydedilir)
    model_sonuclari.append({
        "Model": isim,
        "R-Kare": r2,
        "Adj R-Kare": adj_r2,
        "MAE ($)": mae,
        "RMSE ($)": rmse,
        "Süre (sn)": round(sure, 1) if isinstance(sure, (int, float)) else sure
    })

# ===========================
#  TABLO VE GÖRSEL ÇIKTILAR 
# ===========================

# 1. Liderlik Tablosu
liderlik_tablosu = pd.DataFrame(model_sonuclari).sort_values(by="R-Kare", ascending=False).reset_index(drop=True)

print("\n --- sonuclar --- ")
format_dict = {'R-Kare': '{:.4f}', 'Adj R-Kare': '{:.4f}', 'MAE ($)': '${:.2f}', 'RMSE ($)': '${:.2f}'}
print(liderlik_tablosu.style.format(format_dict).to_string() if hasattr(liderlik_tablosu, 'style') else liderlik_tablosu.to_string())

# 2. Değişken Önem Derecesi (XGBoost Feature Importance)
feature_importances = pd.Series(best_xgb_model.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
feature_importances.nlargest(20).plot(kind='barh', color='#4a235a')
plt.title("Ev Fiyatını Belirleyen En Önemli 20 Faktör (XGBoost)", fontsize=14, fontweight='bold')
plt.xlabel("Modele Etkisi (%)", fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 3. Model Karşılaştırma Grafiği
plt.figure(figsize=(10, 5))
ax = sns.barplot(x='R-Kare', y='Model', data=liderlik_tablosu, palette='magma')
plt.title("Hangi Algoritma Daha Başarılı? (R-Kare Karşılaştırması)", fontsize=14, fontweight='bold')
plt.xlabel("R-Kare Skoru", fontsize=12)
plt.ylabel("Algoritmalar", fontsize=12)
plt.xlim(0, 1) 

for index, value in enumerate(liderlik_tablosu['R-Kare']):
    plt.text(value + 0.01, index, f"{value:.4f}", va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()