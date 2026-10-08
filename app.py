from flask import Flask, render_template, request, url_for, redirect, url_for
import os
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model
from ultralytics import YOLO
import re
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load YOLO
yolo_model = YOLO('best.pt')

# Load classifier
model = load_model('Models/densenet121.h5', compile=False)

class_names = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Bacterial Blight in cotton', 'Becterial Blight in Rice', 'Brownspot',
    'Common_Rust', 'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot',
    'Healthy Maize', 'Healthy Wheat', 'Healthy cotton', 'Leaf Curl', 'Leaf smut',
    'Mosaic sugarcane', 'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast',
    'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly',
    'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite',
    'Wheat powdery mildew', 'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt',
    'Yellow Rust Sugarcane', 'bollrot on Cotton', 'bollworm on Cotton',
    'cotton mealy bug', 'cotton whitefly', 'maize ear rot',
    'maize fall armyworm', 'maize stem borer', 'pink bollworm in cotton',
    'red cotton bug', 'thirps on cotton'
]

# --- Find Last Conv Layer ---
last_conv = None
for layer in reversed(model.layers):
    if len(layer.output_shape) == 4:
        last_conv = layer.name
        break

def make_gradcam(model, img_tensor, class_idx):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0].numpy()
    pooled = pooled.numpy()

    for i in range(conv_out.shape[-1]):
        conv_out[:, :, i] *= pooled[i]

    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap



# ------------------------------------------------
# CLASSIFICATION ROUTE
# ------------------------------------------------
@app.route("/classify", methods=["POST"])
def classify():
    file = request.files.get("classify_image")

    if not file or not allowed_file(file.filename):
        return render_template("home.html", error="Invalid Classification Image")

    filename = "classify_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocessing
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (128, 128))
    arr = img_to_array(img_resized)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    prediction = class_names[idx]

    # GradCAM
    heatmap = make_gradcam(model, arr, idx)
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join(OUTPUT_FOLDER, "gradcam_" + filename)
    cv2.imwrite(output_path, overlay)

    return render_template(
        "result.html",
        result_type="classification",
        prediction=prediction,
        original_url=url_for("static", filename="uploads/" + filename),
        gradcam_url=url_for("static", filename="output/" + "gradcam_" + filename)
    )



# ------------------------------------------------
# DETECTION ROUTE
# ------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("detect_image")

    if not file or not allowed_file(file.filename):
        return render_template("home.html", error="Invalid Detection Image")

    filename = "detect_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = yolo_model(filepath)
    output_path = os.path.join(OUTPUT_FOLDER, "detect_" + filename)

    result_img = results[0].plot()
    cv2.imwrite(output_path, result_img)

    return render_template(
        "result.html",
        result_type="detection",
        image_url=url_for("static", filename="output/" + "detect_" + filename)
    )



@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')


@app.route("/graphs1")
def graphs1():
    return render_template("graphs1.html")


@app.route("/graphs2")
def graphs2():
    return render_template("graphs2.html")


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



if __name__ == "__main__":
    app.run()
