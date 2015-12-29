import sys
import os

# Add the Test Folder path to the sys.path list

path = os.path.join(os.path.dirname(__file__), '..') 
print(path) 
sys.path.append(path)
