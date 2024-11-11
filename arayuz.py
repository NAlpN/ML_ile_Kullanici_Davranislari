import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_os():
    try:
        age = int(age_entry.get())
        gender = gender_var.get()
        
        gender_encoded = 0 if gender == "Female" else 1
        
        input_data = np.array([[age, gender_encoded]])
        
        prediction = model.predict(input_data)[0]
        
        messagebox.showinfo("Tahmin Sonucu", f"Önerilen İşletim Sistemi: {prediction}")
    
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir yaş giriniz.")

root = tk.Tk()
root.title("İşletim Sistemi Öneri Uygulaması")

tk.Label(root, text="Yaşınızı Girin:").grid(row=0, column=0, padx=10, pady=10)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Cinsiyetinizi Seçin:").grid(row=1, column=0, padx=10, pady=10)
gender_var = tk.StringVar(value="Female")  # Varsayılan olarak Female
tk.Radiobutton(root, text="Kadın", variable=gender_var, value="Female").grid(row=1, column=1)
tk.Radiobutton(root, text="Erkek", variable=gender_var, value="Male").grid(row=1, column=2)

predict_button = tk.Button(root, text="İşletim Sistemi Öner", command=predict_os)
predict_button.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()