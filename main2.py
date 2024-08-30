'''
Agent testing
'''
import dotenv

dotenv.load_dotenv()
# import os
# def ensure_directory_exists(directory_path):
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
# ensure_directory_exists("static")     

from persona_private.frontend import app



if __name__ == "__main__":
    app.run()