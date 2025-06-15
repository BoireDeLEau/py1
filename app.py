from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms
from PIL import Image
import torch
from ocr import OCRModel
from data import idx2char, char2idx
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_very_secret_key_here' 

def load_ic_mapping(file_path):
    ic_dict = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): 
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    ic_dict[key.strip().upper()] = value.strip()
    except Exception as e:
        print(f"Error loading IC mapping: {str(e)}")
    return ic_dict

IC_FILE = "photo/IC.txt"
IC = load_ic_mapping(IC_FILE)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制16MB

def load_users():
    users = {}
    try:
        with open('users.txt', 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        username = parts[0].strip()
                        stored_hash = parts[2].strip()  # 哈希验证
                        users[username] = stored_hash
    except FileNotFoundError:
        pass
    return users

def save_user(username, password):
    with open('users.txt', 'a', encoding='utf-8') as f:
        line = f"{username} | {password} | {generate_password_hash(password)}\n"
        f.write(line)

def ctc_decode(preds, idx2char):
    blank = 0
    prev = blank
    result = []
    for p in preds:
        if p != blank and p != prev:
            result.append(idx2char[p])
        prev = p
    return ''.join(result)

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def predict(image_path, model_path="ocr_best.pt"):
    device = torch.device("cuda" if torch.cpu.is_available() else "cpu")

    # 加载模型
    num_classes = len(char2idx)
    model = OCRModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载图像
    img_tensor = load_image(image_path).to(device)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)  # [T, B, C]
        log_probs = output.log_softmax(2)
        preds = log_probs.argmax(2).squeeze(1).cpu().numpy()  # [T]

    # 解码
    pred_text = ctc_decode(preds, idx2char)
    return pred_text

def process_filename(filename):
    try:
        match = re.match(r'^([A-Za-z0-9]+)', filename.upper())
        if not match:
            return None
        base_name = match.group(1)
        return IC.get(base_name, None)
    except Exception as e:
        print(f"Error processing filename: {str(e)}")
        return None

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            result = process_filename(filename)
            
            if result:
                return jsonify({
                    'original': filename,
                    'result': result
                })
            return jsonify({'error': 'Filename pattern not recognized'}), 400
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error='无效的用户名或密码')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            return render_template('register.html', error='用户名和密码不能为空')
            
        users = load_users()
        if username in users:
            return render_template('register.html', error='用户名已存在')
            
        save_user(username, password)
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)