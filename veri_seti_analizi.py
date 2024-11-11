import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('user_behavior_dataset.csv')

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Age', hue='Operating System', palette='viridis')
plt.title("Yaşa Göre Kullanılan İşletim Sistemleri")
plt.xlabel("Yaş")
plt.ylabel("Kullanıcı Sayısı")
plt.legend(title="İşletim Sistemi")
plt.savefig('yas-isletimSistemi.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Gender', y='Screen On Time (hours/day)', palette='magma')
plt.title("Cinsiyete Göre Ekrana Bakma Süreleri")
plt.xlabel("Cinsiyet")
plt.ylabel("Ekran Süresi (saat/gün)")
plt.savefig('cinsiyet-ekranSuresi.png')

device_counts = data['Device Model'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Cihaz Modeli Kullanım Kıyaslamaları")
plt.savefig('cihazKullanimKiyaslamasi.png')