from flask import Flask, render_template, request
from openai import OpenAI

client = OpenAI()
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""

    if request.method == "POST":
        user_prompt = request.form["prompt"]
        temperature = float(request.form["temperature"])
        top_p = float(request.form["top_p"])
        top_k = int(request.form["top_k"])
        cot = request.form.get("cot") == "on"

        system_msg = "Think step by step before responding clearly." if cot else "Respond directly & concisely."

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        response_text = response.choices[0].message.content

    return render_template("index.html", response=response_text)


if __name__ == "__main__":
    app.run(debug=True)