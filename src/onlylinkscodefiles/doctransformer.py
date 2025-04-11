import os
import json
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from src.modules.linkhandler import handle_links
from src.modules.jsoncleaner import cleaner
from src.modules.chunker import chunker


def flatten_documents(doc_list):
    flat_list = []
    for item in doc_list:
        if isinstance(item, list):  # If it's a list, recursively flatten it
            flat_list.extend(flatten_documents(item))
        elif isinstance(item, Document):  # If it's a Document, add it to the list
            flat_list.append(item)
        else:
            raise TypeError(f"Unexpected type: {type(item)}. Expected Document or list of Documents.")
    return flat_list


def convert_to_langchain_documents(base_path, parent_website):
    documents = []
    parent_folder = os.path.join(base_path, parent_website)
    
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"Parent website folder '{parent_folder}' does not exist.")
    
    json_dir = os.path.join(parent_folder, "json")
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        
        json_dict = json.loads(file_content)

        relation_type = "parent" if idx == 0 else "child"
        t = json_dict.get("text", "")
        c = json_dict.get("code_blocks", None)
        if len(str(t)):
            documents.append(Document(
                page_content=t, 
                metadata={"source": parent_website, "type": "text", "relation": relation_type}
            ))
            # print(documents[-1].page_content[:2000])
        
        if c:
            for code_block in c:
                documents.append(Document(
                    page_content=code_block["content"],
                    metadata={"source": parent_website, "type": "code", "language": code_block["language"], "relation": relation_type}
                ))
                # print(documents[-1].page_content)
    
    image_dir = os.path.join(parent_folder, "images")
    if os.path.exists(image_dir):
        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            loader = TextLoader(img_path, encoding='utf-8')
            extracted_doc = loader.load()
            for doc in extracted_doc: 
                doc.metadata["source"] = parent_website
                doc.metadata["type"] = "image"
                doc.metadata["relation"] = "child"
            documents.append(extracted_doc)
    
    tables_dir = os.path.join(parent_folder, "tables")
    if os.path.exists(tables_dir):
        for table_file in os.listdir(tables_dir):
            table_path = os.path.join(tables_dir, table_file)
            table_df = pd.read_csv(table_path)
            table_json = table_df.to_json(orient="records")
            documents.append(Document(
                page_content=table_json,
                metadata={"source": parent_website, "type": "table", "relation": "child"}
            ))
    
    return documents


def transformation_pipeline(urls, depth=1, num_links=2, chunk_size=1000, chunk_overlap=200):
    print("\ninitiating doctransformer: `transformation_pipeline`")

    print("\ntransfer control to linkhandler: `handle_links`")
    directory_names = handle_links(urls, depth, num_links)
    # directory_names = ["httpsenwikipediaorgwikigenv"] # testing purpose

    print("\ntransfer control to jsoncleaner: `cleaner`")
    cleaner()
    base_path = os.path.join("artifacts", "cleaned")
    final_docs_lc = []
    final_docs_dict = {}

    print("\nintiating chunker: `chunker`")
    for dir_name in directory_names:
        print(f"converting to langchain documents: {dir_name}")
        langchain_docs = convert_to_langchain_documents(base_path, dir_name)
        
        flat_langchain_docs = flatten_documents(langchain_docs)
        # print(f"total flatten documents: {len(flat_langchain_docs)}")
        
        chunked_docs = chunker(flat_langchain_docs, chunk_size, chunk_overlap)
        final_docs_dict[dir_name] = chunked_docs
        final_docs_lc.extend(chunked_docs)
    
    print(f"\ntotal link transformed: {len(final_docs_dict.keys())}")
    # testing purpose only
    # for i, doc in enumerate(final_docs_dict):
    #     print(f"\nwebsite | {list(final_docs_dict.keys())[i]}")
    #     print(f"documents count: {len(list(final_docs_dict.values())[i])}")
    print(f"total document count: {len(final_docs_lc)}")
    
    return final_docs_dict, final_docs_lc


# local testing
# urls = [
#         "https://en.wikipedia.org/wiki/Gen_V"
#     ]
# _, result = transformation_pipeline(urls)
# print("------------logging--------------")
# print(f"length of result: {len(result)}")
# print(f"type of result: {type(result)}")

# for r in result:
#     print(type(r))
#     print(r.page_content[:200])
#     print(len(r.page_content))
#     print(r.metadata)