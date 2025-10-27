# backend/app.py
import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import jwt
import datetime

# 設定（本番では環境変数で設定）
PASSWORD = os.environ.get("PASSWORD", "sora0419")  # ← デフォルトは要求された値
SECRET_KEY = os.environ.get("SECRET_KEY", "change_me_please")
JWT_EXP_MINUTES = int(os.environ.get("JWT_EXP_MINUTES", "15"))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 適宜制限してください

def create_token():
    payload = {
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXP_MINUTES),
        "iat": datetime.datetime.utcnow(),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    # pyjwt >=2.0 returns str; ensure str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True
    except Exception:
        return False

@app.route("/auth", methods=["POST"])
def auth():
    data = request.get_json() or {}
    pw = data.get("password")
    if pw is None:
        return jsonify({"ok": False, "error": "password required"}), 400
    if pw == PASSWORD:
        token = create_token()
        return jsonify({"ok": True, "token": token})
    else:
        return jsonify({"ok": False, "error": "invalid password"}), 401

def read_image_file(file_storage):
    file_bytes = file_storage.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

@app.route("/merge", methods=["POST"])
def merge_images():
    # Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False, "error": "missing token"}), 401
    token = auth_header.split(" ", 1)[1]
    if not verify_token(token):
        return jsonify({"ok": False, "error": "invalid or expired token"}), 401

    if 'imageA' not in request.files or 'imageB' not in request.files:
        return jsonify({"ok": False, "error": "imageA and imageB required"}), 400

    fA = request.files['imageA']
    fB = request.files['imageB']

    A = read_image_file(fA)
    B = read_image_file(fB)
    if A is None or B is None:
        return jsonify({"ok": False, "error": "could not decode images"}), 400

    # resize A to B size
    h, w = B.shape[:2]
    A_resized = cv2.resize(A, (w, h))

    # Low-frequency (A) - Gaussian blur
    # カーネルサイズは画像サイズ次第で調整可
    k = max(15, int(min(w,h) / 20))  # 自動的な基準
    if k % 2 == 0: k += 1
    low = cv2.GaussianBlur(A_resized, (k, k), 0)

    # High-frequency (B) - blur then subtract (approx highpass)
    k2 = max(7, int(min(w,h) / 50))
    if k2 % 2 == 0: k2 += 1
    blurB = cv2.GaussianBlur(B, (k2, k2), 0)
    # high = B - blurB scaled; but use addWeighted for stability
    high = cv2.addWeighted(B, 1.5, blurB, -0.5, 0)

    # 合成比率（必要なら動的に変えられる）
    alpha_low = float(request.form.get("alpha_low", 0.7))  # 0.0 - 1.0
    alpha_high = float(request.form.get("alpha_high", 0.3))

    C = cv2.addWeighted(low, alpha_low, high, alpha_high, 0)

    # JPEG 出力
    _, buf = cv2.imencode(".jpg", C, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    bytes_io = io.BytesIO(buf.tobytes())
    bytes_io.seek(0)
    return send_file(bytes_io, mimetype="image/jpeg", as_attachment=False, attachment_filename="C.jpg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
