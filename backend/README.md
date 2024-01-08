To run backend: 
- For linux: 
  - pip install virtualenv 
  - virtualenv env 
  - source env/bin/activate 
  - pip install -r requirements.txt

- For windows:
  - python -m pip install virtualenv 
  - python -m virtualenv env 
  - .\env\Scripts\activate


- run `uvicorn app.main:app --reload --host 0.0.0.0 --port 3003` to start server on port 3003
