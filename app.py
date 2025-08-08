from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        mean_radius = float(request.form["mean_radius"])
        mean_texture = float(request.form["mean_texture"])
        mean_perimeter = float(request.form["mean_perimeter"])
        mean_area = float(request.form["mean_area"])

        input_data = [0] * model.n_features_in_
        input_data[0] = mean_radius
        input_data[1] = mean_texture
        input_data[2] = mean_perimeter
        input_data[3] = mean_area

        prediction = model.predict([input_data])[0]
        result = "Malignant" if prediction == 0 else "Benign"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
