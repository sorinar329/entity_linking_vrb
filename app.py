import os
import sys
from flask import Flask, render_template, request, jsonify, url_for

# =============================
# Setup project paths
# =============================
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

import scripts.pipeline as pipeline
import scripts.cutting_queries as cutting

# =============================
# Flask setup
# =============================
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =============================
# Main page
# =============================
@app.route("/")
def index():
    return render_template("index.html")


# =============================
# Upload image
# =============================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    return jsonify({
        "filename": file.filename,
        "url": url_for("static", filename="uploads/" + file.filename)
    })


# =============================
# Use sample image
# =============================
@app.route("/use_sample")
def use_sample():
    filename = "multipleFruits.png"

    return jsonify({
        "filename": filename,
        "url": url_for("static", filename="uploads/" + filename)
    })


# =============================
# Detect objects (bounding boxes)
# =============================
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    filename = data["filename"]

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    boxes = pipeline.get_boxes_only(img_path)

    return jsonify({"boxes": boxes})

# =============================
# Handle click + full pipeline
# =============================
@app.route("/click", methods=["POST"])
def click():
    data = request.get_json()

    x = data["x"]
    y = data["y"]
    filename = data["filename"]

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # detect boxes
    results = pipeline.get_bboxes(img_path)

    boxes = []
    for r in results:
        for b in r["boxes"]:
            boxes.append(b.tolist())

    # find selected box
    selected_box = None
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            selected_box = box
            break

    # run your original logic
    clicked_obj = pipeline.get_clicked_obj(img_path, x, y)

    verb = "cut:Quartering"
    steps = cutting.build_motion_table(clicked_obj[0], verb)

    return jsonify({
        "steps": steps.to_dict(orient="records"),
        "selected_box": selected_box,
        "boxes": boxes
    })


# =============================
# Run
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)