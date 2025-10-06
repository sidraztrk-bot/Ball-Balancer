import numpy as np
import cv2
import time
import serial

class STM32BallTracker:
    def __init__(self, camera_source=0, stm32_port='COM9', baudrate=38400, 
                 real_width_cm=14, real_height_cm=14):
        """
        STM32 Ball Tracker - ROI seçimi + CM dönüşüm + Turuncu top algılama
        
        Args:
            real_width_cm: ROI'nin gerçek dünya genişliği (cm)
            real_height_cm: ROI'nin gerçek dünya yüksekliği (cm)
        """
        self.camera_source = camera_source
        self.stm32_port = stm32_port
        self.baudrate = baudrate
        self.ser = None

        # ROI parametreleri
        self.roi = None  # (x, y, w, h)
        self.roi_selected = False
        self.real_width_cm = real_width_cm
        self.real_height_cm = real_height_cm
        
        # Ball detection parametreleri - turuncu top için optimize
        self.min_area = 300      # Daha küçük toplar için düşürüldü
        self.max_area = 50000     
        self.min_circularity = 0.4  # Daha toleranslı
        
        # Beyaz renk HSV aralıkları
        # Hangi beyaz tonunu istiyorsan değerleri daraltabiliriz
        self.lower_white = np.array([0, 0, 120])     
        self.upper_white = np.array([180, 80, 255])
        
        self.frame_center = None
        self.running = False
        self.last_sent_coords = None

    def init_stm32_connection(self):
        try:
            self.ser = serial.Serial(self.stm32_port, self.baudrate, timeout=1)
            print(f"✅ Connected to STM32 on {self.stm32_port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to STM32: {e}")
            return False

    def select_roi(self, frame):
        """İlk frame'de manuel ROI seçimi"""
        print("\n📐 ROI SELECTION MODE")
        print("=" * 50)
        print("Instructions:")
        print("  1. Click and drag to select the tracking area")
        print("  2. Press ENTER to confirm selection")
        print("  3. Press 'c' to cancel and reselect")
        print("=" * 50)
        
        # ROI seçimi
        roi = cv2.selectROI("Select Tracking Area (Press ENTER)", frame, 
                           fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Tracking Area (Press ENTER)")
        
        if roi[2] > 0 and roi[3] > 0:  # Geçerli seçim yapıldı mı?
            self.roi = roi
            self.roi_selected = True
            
            x, y, w, h = roi
            print(f"\n✅ ROI Selected:")
            print(f"   Position: ({x}, {y})")
            print(f"   Size: {w}x{h} pixels")
            print(f"   Real world: {self.real_width_cm}x{self.real_height_cm} cm")
            print(f"   Scale: {w/self.real_width_cm:.2f} pixels/cm")
            
            # ROI merkez noktası
            self.frame_center = (x + w//2, y + h//2)
            print(f"   Center: {self.frame_center}")
            
            return True
        else:
            print("❌ No valid ROI selected!")
            return False

    def pixel_to_cm(self, pixel_x, pixel_y):
        """Piksel koordinatlarını cm'ye dönüştür (ROI içinde)"""
        if not self.roi_selected:
            return None
        
        roi_x, roi_y, roi_w, roi_h = self.roi
        
        # ROI içindeki relatif pozisyon (0-1 arası)
        rel_x = (pixel_x - roi_x) / roi_w
        rel_y = (pixel_y - roi_y) / roi_h
        
        # CM'ye dönüştür (sol üst köşe origin)
        cm_x = rel_x * self.real_width_cm
        cm_y = rel_y * self.real_height_cm
        
        # Merkez origin'e dönüştür (isteğe bağlı)
        cm_x_centered = cm_x - (self.real_width_cm / 2)
        cm_y_centered = cm_y - (self.real_height_cm / 2)
        
        return {
            'pixel': (pixel_x, pixel_y),
            'cm_absolute': (round(cm_x, 2), round(cm_y, 2)),
            'cm_centered': (round(cm_x_centered, 2), round(cm_y_centered, 2))
        }

    def send_coordinates_to_stm32(self, coords_dict):
        """CM koordinatlarını STM32'ye gönder"""
        if self.ser and self.ser.is_open and coords_dict:
            try:
                # Merkez tabanlı koordinatları gönder
                cm_x, cm_y = coords_dict['cm_absolute']

                cm_x_int = int(cm_x * 100)
                cm_y_int = int(cm_y * 100)

                msg = f"{cm_x_int},{cm_y_int}\n"
                self.ser.write(msg.encode())

                print(f"📤 Sent to STM32 (x100): {msg.strip()}")

            except Exception as e:
                print(f"❌ Error sending data: {e}")

    def detect_ball(self, frame):
        """Turuncu top algılama (sadece ROI içinde)"""
        height, width = frame.shape[:2]
        
        # ROI seçilmemişse tam frame'i kullan
        if not self.roi_selected:
            search_frame = frame
            roi_offset = (0, 0)
        else:
            # Sadece ROI bölgesini işle
            x, y, w, h = self.roi
            search_frame = frame[y:y+h, x:x+w]
            roi_offset = (x, y)
        
        # HSV dönüşümü
        hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        
        # Gürültü temizleme - daha agresif
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Gaussian blur - daha yumuşak kenarlar
        white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
        
        # Contour bulma
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_center = None
        best_contour = None
        coords_dict = None
        
        # En uygun contour'u bul
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.min_circularity:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"]) + roi_offset[0]
                            cy = int(M["m01"] / M["m00"]) + roi_offset[1]
                            ball_center = (cx, cy)
                            
                            # Global koordinatlara çevir (contour lokal)
                            global_contour = contour.copy()
                            global_contour[:, :, 0] += roi_offset[0]
                            global_contour[:, :, 1] += roi_offset[1]
                            best_contour = global_contour
                            
                            # CM'ye dönüştür
                            coords_dict = self.pixel_to_cm(cx, cy)
                            break
        
        # ROI çiz
        if self.roi_selected:
            x, y, w, h = self.roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, f"ROI: {self.real_width_cm}x{self.real_height_cm}cm", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Top görselleştirmesi
        if ball_center and best_contour is not None and coords_dict:
            # Top merkezi ve çevre
            cv2.circle(frame, ball_center, 6, (0, 255, 0), -1)
            cv2.circle(frame, ball_center, 25, (0, 255, 0), 2)
            cv2.drawContours(frame, [best_contour], -1, (255, 0, 0), 2)
            
            # Koordinat bilgileri
            px, py = coords_dict['pixel']
            cm_abs_x, cm_abs_y = coords_dict['cm_absolute']
            cm_cen_x, cm_cen_y = coords_dict['cm_centered']
            
            # Çoklu satır bilgi
            info_lines = [
                f"Pixel: ({px}, {py})",
                f"CM: ({cm_abs_x}, {cm_abs_y})",
                f"Centered: ({cm_cen_x}, {cm_cen_y})"
            ]
            
            y_offset = ball_center[1] - 60
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (ball_center[0] + 30, y_offset + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Merkez noktası
        if self.frame_center:
            cv2.circle(frame, self.frame_center, 8, (0, 0, 255), -1)
            cv2.line(frame, (self.frame_center[0]-15, self.frame_center[1]), 
                    (self.frame_center[0]+15, self.frame_center[1]), (0, 0, 255), 2)
            cv2.line(frame, (self.frame_center[0], self.frame_center[1]-15), 
                    (self.frame_center[0], self.frame_center[1]+15), (0, 0, 255), 2)
        
        return frame, white_mask, coords_dict

    def run(self):
        """Ana döngü"""
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print(f"❌ Camera connection failed: {self.camera_source}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera connected: {self.camera_source}")
        
        # İlk frame'i al ve ROI seç
        ret, first_frame = cap.read()
        if not ret:
            print("❌ Could not read first frame!")
            cap.release()
            return False
        
        # ROI seçimi
        if not self.select_roi(first_frame):
            cap.release()
            return False
        
        # STM32 bağlantısı
        stm32_connected = self.init_stm32_connection()
        
        self.running = True
        print("\n🎯 STM32 Ball Tracker Started")
        print("Controls: Q - Quit | R - Reselect ROI | S - Toggle STM32")
        print("=" * 50)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read error")
                    break
                
                tracked_frame, mask, coords_dict = self.detect_ball(frame.copy())
                
                # STM32'ye gönder
                if stm32_connected and coords_dict:
                    self.send_coordinates_to_stm32(coords_dict)
                
                cv2.imshow('STM32 Ball Tracker', tracked_frame)
                cv2.imshow('Orange Mask', mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Stopped by user")
                    break
                elif key == ord('r'):
                    # ROI yeniden seçimi
                    if self.select_roi(frame):
                        print("✅ ROI reselected")
                elif key == ord('s'):
                    # STM32 toggle
                    if stm32_connected:
                        print("🔌 STM32 sending disabled")
                        stm32_connected = False
                    else:
                        print("📡 STM32 sending enabled")
                        stm32_connected = True
        
        except KeyboardInterrupt:
            print("\n⏹ Stopped with Ctrl+C")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("🔌 STM32 connection closed")
            print("✅ System cleaned up")
        
        return True


def main():
    print("🤖 STM32 Ball Tracker - ROI + CM Conversion")
    print("=" * 50)
    
    # Konfigürasyon
    CAMERA_SOURCE = "http://192.168.1.40:8080/video"  # IP kamera
    # CAMERA_SOURCE = 0  # Webcam
    
    STM32_PORT = 'COM9'
    BAUDRATE = 38400
    
    REAL_WIDTH_CM = 14   # Gerçek dünya genişliği
    REAL_HEIGHT_CM = 14  # Gerçek dünya yüksekliği
    
    print(f"📱 Camera: {CAMERA_SOURCE}")
    print(f"🔌 STM32: {STM32_PORT} @ {BAUDRATE}")
    print(f"📏 Real world size: {REAL_WIDTH_CM}x{REAL_HEIGHT_CM} cm")
    print("-" * 50)
    
    tracker = STM32BallTracker(
        camera_source=CAMERA_SOURCE,
        stm32_port=STM32_PORT,
        baudrate=BAUDRATE,
        real_width_cm=REAL_WIDTH_CM,
        real_height_cm=REAL_HEIGHT_CM
    )
    
    success = tracker.run()
    if success:
        print("✅ Program completed successfully")
    else:
        print("❌ Program ended with error")


if __name__ == "__main__":
    main()