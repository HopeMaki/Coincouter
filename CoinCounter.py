import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import sys
import os

# --- ฟังก์ชันสำหรับหา Path เมื่อกลายเป็นไฟล์ .exe ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- ตอนโหลดโมเดล ให้ใช้ฟังก์ชันนี้เรียก ---
model_path = resource_path('best.pt')
model = YOLO(model_path)

class CoinApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Coin Counter")
        
        # Load Model
        self.model = YOLO('best.pt')
        
        # UI Setup
        # ฝั่งซ้าย: แสดงภาพจากกล้อง
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(side=tk.LEFT)
        
        # ฝั่งขวา: แสดง Log และยอดรวม
        self.sidebar = tk.Frame(window, padx=20)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.total_label = tk.Label(self.sidebar, text="Total: 0 Baht", font=("Arial", 24, "bold"), fg="green")
        self.total_label.pack(pady=20)
        
        self.info_label = tk.Label(self.sidebar, text="Detected Coins:", font=("Arial", 12, "bold"))
        self.info_label.pack(anchor="w")
        
        self.log_text = tk.Text(self.sidebar, width=30, height=15, font=("Consolas", 10))
        self.log_text.pack()
        
        # เปิดกล้อง
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.is_running = True
        self.update_frame()
        
    def update_frame(self):
        ret, img = self.cap.read()
        if ret and self.is_running:
            img = cv2.flip(img, 1)
            # รัน Model
            results = self.model(img, device='cpu', verbose=False)
            
            total = 0
            coin_counts = {}
            
            for result in results:
                for boxes in result.boxes:
                    # ดึงพิกัดและข้อมูลเหรียญ 
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
                    class_id = int(boxes.cls[0])
                    conf = float(boxes.conf[0]*100)
                    name = result.names[class_id]
                    
                    # วาดกรอบสี่เหลี่ยมและ Label ลงบนภาพ (เหมือนโค้ดเดิมที่กี้ชอบ) 
                    label_str = f"{name} {conf:.1f}%"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # พื้นหลังข้อความ Label
                    (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 0, 0), -1)
                    cv2.putText(img, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Logic การนับเงิน
                    if "10baht" in name: total += 10
                    elif "5baht" in name: total += 5
                    elif "2baht" in name: total += 2
                    elif "1baht" in name: total += 1
                    
                    coin_counts[name] = coin_counts.get(name, 0) + 1
            
            # อัปเดตยอดเงินใน GUI
            self.total_label.config(text=f"Total: {total} Baht")
            
            # อัปเดตรายการเหรียญใน Log
            self.log_text.delete('1.0', tk.END)
            if coin_counts:
                for c_name, count in sorted(coin_counts.items()):
                    self.log_text.insert(tk.END, f"• {c_name}: {count} เหรียญ\n")
            else:
                self.log_text.insert(tk.END, "ไม่พบเหรียญในขณะนี้")
            
            # แปลงภาพ OpenCV (BGR) เป็น RGB เพื่อโชว์ใน Tkinter
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.canvas.image = img_tk
            
        self.window.after(10, self.update_frame)

    def on_closing(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CoinApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()