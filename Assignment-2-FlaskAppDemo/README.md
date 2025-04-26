# Assignment 02 of 10, Simple Flask App Demonstration.
## Simple Flask "Hello World" App 👋

A minimal example of a web application built using the Python [Flask](https://flask.palletsprojects.com/) framework. This app demonstrates the basic structure and serves a single page displaying "Hello, World!".

---

## ✨ Features

*   Displays "Hello, World!" on the homepage (`/`).
*   Uses Flask's built-in development server.
*   Includes debug mode for easier development.

---

## 🛠️ Tech Stack

*   **Python:** Core programming language.
*   **Flask:** Micro web framework for Python.

---

## ⚙️ Setup and Installation

1.  **Prerequisites:** Ensure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/).

2.  **Clone Repository (Optional):** If this code is part of a Git repository, clone it:
    ```bash
    git clone [<repository_url>](https://github.com/thilak-r/machine-learning-assignments/edit/main/Assingment-2-FlaskAppDemo/app.py
    cd Assingment-2-FlaskAppDemo
    ```
    Otherwise, simply save the `app.py` code into a directory.

3.  **Create a Virtual Environment (Recommended):** It's good practice to create a virtual environment to manage project dependencies separately.
    ```bash
    # Create a virtual environment named 'venv'
    python -m venv venv

    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies:** Install Flask using pip.
    ```bash
    pip install Flask
    ```

---

## ▶️ Running the Application

1.  Make sure your virtual environment is activated (you should see `(venv)` before your terminal prompt).
2.  Navigate to the directory containing `app.py`.
3.  Run the Flask development server using the Python interpreter:
    ```bash
    python app.py
    ```
4.  You should see output similar to this in your terminal, indicating the server is running:
    ```
     * Serving Flask app 'app'
     * Debug mode: on
     * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
     * Restarting with stat
     * Debugger is active!
     * Debugger PIN: xxx-xxx-xxx
    ```
5.  Open your web browser and navigate to the URL provided:
    **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**
    *(Note: The port `5000` is the default for Flask's development server)*

6.  You should see the text **"Hello, World!"** displayed on the web page.

---

## 📝 Code Explanation (`app.py`)

*   `from flask import Flask`: Imports the necessary `Flask` class from the library.
*   `app = Flask(__name__)`: Creates an instance of the Flask application. `__name__` helps Flask determine the root path for the application.
*   `@app.route('/')`: This is a decorator that defines a *route*. It maps the URL path `/` (the root/homepage) to the function defined immediately below it (`hello_world`).
*   `def hello_world():`: This is the *view function* that gets executed when a user accesses the `/` route.
*   `return 'Hello, World!'`: The view function returns the content (in this case, a simple string) that will be sent back to the user's browser.
*   `if __name__ == '__main__':`: This standard Python construct ensures that the following code only runs when the script is executed directly (not when it's imported as a module).
*   `app.run(debug=True)`: Starts Flask's built-in development web server. `debug=True` enables helpful features for development, like automatic reloading when code changes and detailed error pages. **Do not use `debug=True` in a production environment!**

---

---

## 📬 Contact

- 👤 **Author**: Thilak R
- 📧 **Email**: [thilak22005@gmail.com](mailto:thilak22005@egmail.com)
- 🌐 **GitHub**: [thilak-r](https://github.com/thilak-r)


  <br><br>
under guidance of [Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu)


