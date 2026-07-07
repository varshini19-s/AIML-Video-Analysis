import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import sys

root = tk.Tk()
root.title("Security System")
root.geometry("500x350")

# ===== Functions to open apps =====
def open_fight_predict():
    subprocess.Popen([sys.executable, "final fight.py"])

def open_thief_predict():
    subprocess.Popen([sys.executable, "Thief.py"])

def open_live_predict():
    subprocess.Popen([sys.executable, "demo.py"])

# ===== Switch to Main Page =====
def show_main_page():
    login_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

# ===== Dummy Login Check =====
def check_login():
    username = user_entry.get()
    password = pass_entry.get()

    if username == "admin" and password == "1234":
        show_main_page()
    else:
        msg_label.config(text="Invalid Login", fg="red")

# ================= LOGIN FRAME =================
login_frame = tk.Frame(root)
login_frame.pack(fill="both", expand=True)

bg_image1 = Image.open("SEC1.jpg")
bg_image1 = bg_image1.resize((500, 350))
bg_photo1 = ImageTk.PhotoImage(bg_image1)

bg_label1 = tk.Label(login_frame, image=bg_photo1)
bg_label1.place(x=0, y=0, relwidth=1, relheight=1)

tk.Label(login_frame, text="LOGIN",
         font=("Arial", 18, "bold"),
         bg="black", fg="white").pack(pady=20)

tk.Label(login_frame, text="Username", bg="black", fg="white").pack()
user_entry = tk.Entry(login_frame)
user_entry.pack(pady=5)

tk.Label(login_frame, text="Password", bg="black", fg="white").pack()
pass_entry = tk.Entry(login_frame, show="*")
pass_entry.pack(pady=5)

tk.Button(login_frame, text="Login",
          bg="#2ecc71", fg="white",
          command=check_login).pack(pady=10)

msg_label = tk.Label(login_frame, text="", bg="black")
msg_label.pack()

# ================= MAIN BUTTON FRAME =================
main_frame = tk.Frame(root)

bg_image2 = Image.open("IMG1.jpg")
bg_image2 = bg_image2.resize((500, 350))
bg_photo2 = ImageTk.PhotoImage(bg_image2)

bg_label2 = tk.Label(main_frame, image=bg_photo2)
bg_label2.place(x=0, y=0, relwidth=1, relheight=1)

tk.Label(main_frame, text="Welcome!",
         font=("Arial", 20, "bold"),
         fg="white", bg="black").pack(pady=20)

tk.Button(main_frame, text="FIGHT PREDICT",
          bg="#ff6b6b", fg="white",
          width=15, command=open_fight_predict).pack(pady=10)

tk.Button(main_frame, text="THIEF PREDICT",
          bg="#2ecc71", fg="white",
          width=15, command=open_thief_predict).pack(pady=10)

tk.Button(main_frame, text="LIVE PREDICT",
          bg="#3498db", fg="white",
          width=15, command=open_live_predict).pack(pady=10)

root.mainloop()
