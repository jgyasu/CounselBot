from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from database import vector_db


retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        """<|system|>
            You are a knowledgeable assistant that provides accurate answers about college counseling and admissions based on the given context. Use the provided context to answer user queries, and if the context doesn't have enough information, say, "I don't have enough information to answer this question."
            <|end|>
            <|context|>
            {context}
            <|end|>
            <|user|>
            {query}
            <|end|>
            <|assistant|>
        """
    ),
)


model_id = "OpenVINO/Phi-3-mini-128k-instruct-int8-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id)


model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
langchain_llm = HuggingFacePipeline(pipeline=model_pipeline)


qa_chain = RetrievalQA.from_chain_type(
    llm=langchain_llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)


query = input("Enter your query: ")
response = qa_chain.invoke(input=query)


answer = response.get("result", "No answer found.")
sources = response.get("source_documents", [])


print(f"Answer: {answer}")
if sources:
    print("\nSources:")
    for source in sources:
        print(source.metadata.get("source", "Unknown source"))
