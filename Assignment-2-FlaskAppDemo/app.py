from flask import Flask

# Create an instance of the Flask class
# __name__ tells Flask where to look for resources like templates and static files.
app = Flask(__name__)

# Define a route for the homepage ('/')
# The @app.route() decorator tells Flask what URL should trigger our function.
@app.route('/')
def hello_world():
  """This function runs when someone visits the root URL ('/')"""
  # Return the string 'Hello, World!' to be displayed in the browser
  return 'Hello, World!'

# Check if the script is executed directly (not imported)
if __name__ == '__main__':
  # Run the Flask development server
  # debug=True allows for automatic reloading and provides error messages in the browser
  app.run(debug=True)
