# Candidate Profiler


## TO-DO :
- [x] enable print tokens utilizations for the conversation
- [x] Use CSV Agent for chat with the entire csv file
- [ ] Add lots of files accepted like GitHub repo, Excel etc...
- [ ] Add free models like vicuna and free embeddings
- [ ] Replace chain of the chatbot by a custom agent for handling more features | memory + vectorstore + custom prompt


### Installation
Clone the repository 

Navigate to the project directory :

`cd CandidateProfiler`


Create a virtual environment :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies in the virtual environment :

`pip install -r requirements.txt`


Launch the service locally :

`streamlit run job_descriptions.py`

#### That's it! The service is now up and running locally.

### When modifying installed libraries
`pip freeze > requirements.txt`

and add manually the package, for streamlit purpose : 
pysqlite3-binary