from flask import Flask, current_app, render_template
app = Flask(__name__)

@app.route('/')
def home():
  # return current_app.send_static_file('index.html')
  return render_template('index.html')
app.run(debug=True)