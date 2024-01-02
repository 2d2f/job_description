from langchain.chains import ConversationalRetrievalChain

#from datachad.backend.deeplake import get_deeplake_vector_store
from src.backend.logging import logger
from src.backend.models import get_model
from src.backend.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT


def get_search_kwargs(options: dict):
    k = int(options["max_tokens"] // options["chunk_size"])
    fetch_k = k * options["k_fetch_k_ratio"]
    search_kwargs = {
        "maximal_marginal_relevance": options["maximal_marginal_relevance"],
        "distance_metric": options["distance_metric"],
        "fetch_k": fetch_k,
        "k": k,
    }
    return search_kwargs


# def get_qa_chain(
#     data_source: str, vector_store_path: str, options: dict, credentials: dict
# ) -> ConversationalRetrievalChain:
#     # create the langchain that will be called to generate responses
#     # vector_store to change !!!
#     vector_store = vector_store(
#         data_source, vector_store_path, options, credentials
#     )
#     retriever = vector_store.as_retriever()
#     search_kwargs = get_search_kwargs(options)
#     retriever.search_kwargs.update(search_kwargs)
#     model = get_model(options, credentials)
#     chain = ConversationalRetrievalChain.from_llm(
#         model,
#         retriever=retriever,
#         chain_type="stuff",
#         verbose=True,
#         # we limit the maximum number of used tokens
#         # to prevent running into the models context window limit
#         max_tokens_limit=options["max_tokens"],
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT},
#         condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#     )
#     logger.info(f"Chain for data source {data_source} and settings {options} build!")
#     return chain