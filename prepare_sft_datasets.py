import json
import os
import re
import random
try:  # pragma: no cover - optional dependency
    import sqlparse  # type: ignore
except Exception:  # pragma: no cover
    sqlparse = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sql_metadata import Parser  # type: ignore
except Exception:  # pragma: no cover
    Parser = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from utils.bridge_content_encoder import get_matched_entries
except Exception:  # pragma: no cover
    def get_matched_entries(*args, **kwargs):
        return None

from utils.db_utils import get_db_schema

# Pyserini is an optional dependency. Importing it pulls in the whole
# PyTorch stack, which might be unavailable in lightweight environments.
# Try to import the LuceneSearcher; if that fails we simply skip the
# content matching step later on.
try:  # pragma: no cover - best effort import
    from pyserini.search.lucene import LuceneSearcher  # type: ignore
except Exception:  # pragma: no cover - if pyserini/torch not installed
    LuceneSearcher = None  # type: ignore

random.seed(42)

def extract_large_numbers(text):
    number_information = []
    patterns = {
        'thousand': 10**3,
        'million': 10**6,
        'billion': 10**9,
        'trillion': 10**12
    }
    
    for word, multiplier in patterns.items():
        matches = re.findall(r'(\d+\.?\d*)\s*{}'.format(word), text, flags=re.IGNORECASE)
        for match in matches:
            number = float(match) * multiplier
            number_information.append(match + " " + word + " = " + str(int(number)))
    
    for phrase, number in {'thousands of': 10**3, 'millions of': 10**6, 'billions of': 10**9, 'trillions of': 10**12}.items():
        if phrase in text:
            number_information.append(phrase + " = " + str(int(number)))
    
    large_number_evidence = ""
    for info in number_information:
        large_number_evidence += info + "; "
    
    return large_number_evidence.strip()

def remove_table_alias(s):
    """Remove table aliases from SQL string when Parser is available."""
    if Parser is None:
        return s
    try:
        tables_aliases = Parser(s).tables_aliases
    except Exception:
        return s

    new_tables_aliases = {}
    for i in range(1, 11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        s = s.replace("AS " + k + " ", "")
        s = s.replace(k, v)

    return s

def remove_similar_comments(names, comments):
    '''
    Remove table (or column) comments that have a high degree of similarity with their names
    
    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments
    
    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments = []
    for name, comment in zip(names, comments):    
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    
    return new_comments

def str_replace_ignore_case(evidence, schema_item_name):
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)

    return evidence

def obtain_n_grams(sequence, max_n):
    """Return all n-grams of ``sequence`` up to length ``max_n``.

    This implementation avoids NLTK dependencies and instead relies solely
    on Python's built-in string handling.
    """
    tokens = sequence.split()
    all_grams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            all_grams.append(" ".join(tokens[i : i + n]))
    return all_grams

def preprocess_evidence(evidence, schema_items):
    if evidence.strip() == "":
        return ""

    evidence = evidence.strip()
    # if evidence does not end with ";", add a ";" char
    if not evidence.endswith(";"):
        evidence += ";"
    
    # lowercase schema items appeared in the evidence
    for table in schema_items:
        if table["table_name"] in evidence.lower():
            evidence = str_replace_ignore_case(evidence, table["table_name"])

        for column_name in table["column_names"]:
            if column_name in evidence.lower():
                evidence = str_replace_ignore_case(evidence, column_name)
    
    evidence = evidence.replace("< =", "<=").replace("> =", ">=")

    return evidence

def spider_style_dataset(
    dataset_path, 
    db_path, 
    db_content_index_path, 
    source, 
    table_json_path,
    use_evidence,
    mode
):
    '''
    Load spider-style dataset
    
    Arguments:
        dataset_path: directory to load the dataset from
        db_path: directory of databases (used for extracting schema, including tables, columns, column contents, and foreign keys)
        db_content_index_path: directory of database content sparse index
        source: source of examples
        table_json_path: directory to load additional database information (used for extracting comments for tables and columns)
        use_evidence: whether to use the additional evidence in the input sequence
    Returns:
        returned_dataset: prepared dataset
    '''
    returned_dataset = []

    dataset = json.load(open(dataset_path))
    additional_db_info = json.load(open(table_json_path))

    db_comments = dict()
    # record comments for tables and columns
    for db_info in additional_db_info:
        comment_dict = dict()

        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]]
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]]
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]]
        
        assert len(column_names) == len(column_comments)
        column_comments = remove_similar_comments(column_names, column_comments)

        table_names = [table_name.lower() for table_name in db_info["table_names_original"]]
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]
        
        assert len(table_names) == len(table_comments)
        table_comments = remove_similar_comments(table_names, table_comments)

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict()
            }
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                # record columns in current table
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][column_name] = column_comment

        db_comments[db_info["db_id"]] = comment_dict

    db_ids = set([data["db_id"] for data in dataset])
    db_id2searcher = {}
    # Build searchers only when LuceneSearcher is available. This keeps the
    # script runnable in minimal environments where Pyserini/Torch is missing.
    if LuceneSearcher is not None:
        for db_id in db_ids:
            index_dir = os.path.join(db_content_index_path, db_id)
            if os.path.exists(index_dir):
                try:  # pragma: no cover - best effort
                    db_id2searcher[db_id] = LuceneSearcher(index_dir)
                except Exception:
                    pass

    db_id2schema = dict()

    for data in dataset:
        sample = {}
        db_id = data["db_id"]
        
        sample["db_id"] = db_id
        sample["db_path"] = os.path.join(db_path, db_id, db_id + ".sqlite")

        if db_id in db_id2schema:
            sample["schema"] = db_id2schema[db_id]
        else:
            db_id2schema[db_id] = get_db_schema(sample["db_path"], db_comments, db_id)
            sample["schema"] = db_id2schema[db_id]

        if "spider-syn" in source:
            sample["question"] = data["SpiderSynQuestion"]
            sample["evidence"] = ""
        elif "bird" in source:
            sample["question"] = data["question"]
            evidence = preprocess_evidence(data["evidence"], sample["schema"]["schema_items"])
            sample["evidence"] = evidence
        elif "bank" in source:
            sample["question"] = data["question"]
            sample["evidence"] = extract_large_numbers(data["question"])
        else:
            sample["question"] = data["question"]
            sample["evidence"] = ""
        
        if "\n" in sample["question"]:
            sample["question"] = sample["question"].replace("\n", " ")
        if "\n" in sample["evidence"]:
            sample["evidence"] = sample["evidence"].replace("\n", " ")
        
        sample["text"] = sample["evidence"] + " " + sample["question"] \
            if use_evidence and sample["evidence"] != "" else sample["question"]

        if mode in ["train", "dev"]:
            sql = data["SQL"] if source in ["bird-dev", "bird-train"] else data["query"]
            if sqlparse is not None:
                formatted_sql = sqlparse.format(
                    sql, keyword_case="upper", identifier_case="lower"
                )
            else:
                formatted_sql = sql
            sample["sql"] = remove_table_alias(formatted_sql)
        elif mode == "test":
            sample["sql"] = ""
        
        sample["table_labels"], sample["column_labels"] = [], []
        try:
            sql_tokens = [token.value for token in Parser(sample["sql"].lower()).tokens]
        except Exception as e:
            sql_tokens = sample["sql"].lower().split()
        
        for table_info in sample["schema"]["schema_items"]:
            if mode in ["train", "dev"]:
                table_name = table_info["table_name"]
                sample["table_labels"].append(1 if table_name in sql_tokens else 0)
                sample["column_labels"].append([1 if column_name in sql_tokens or table_name+"."+column_name in sql_tokens else 0 \
                    for column_name in table_info["column_names"]])
            elif mode == "test":
                sample["table_labels"].append(0)
                sample["column_labels"].append([0 for _ in range(len(table_info["column_names"]))])

        fine_matched_contents = {}
        if db_id in db_id2searcher:
            # coarse-grained matching between the input text and all contents in database
            grams = obtain_n_grams(sample["text"], 4)
            hits = []
            searcher = db_id2searcher[db_id]
            for query in grams:
                hits.extend(searcher.search(query, k=10))

            coarse_matched_contents = {}
            for i in range(len(hits)):
                matched_result = json.loads(hits[i].raw)
                tc_name = ".".join(matched_result["id"].split("-**-")[:2])
                if tc_name in coarse_matched_contents.keys():
                    if matched_result["contents"] not in coarse_matched_contents[tc_name]:
                        coarse_matched_contents[tc_name].append(matched_result["contents"])
                else:
                    coarse_matched_contents[tc_name] = [matched_result["contents"]]

            for tc_name, contents in coarse_matched_contents.items():
                fm_contents = get_matched_entries(sample["text"], contents)
                if fm_contents is None:
                    continue
                for _match_str, (
                    field_value,
                    _s_match_str,
                    match_score,
                    s_match_score,
                    _match_size,
                ) in fm_contents:
                    if match_score < 0.9:
                        continue
                    if tc_name in fine_matched_contents.keys():
                        if len(fine_matched_contents[tc_name]) < 25:
                            fine_matched_contents[tc_name].append(field_value.strip())
                    else:
                        fine_matched_contents[tc_name] = [field_value.strip()]

        sample["matched_contents"] = fine_matched_contents
        sample["source"] = source

        returned_dataset.append(sample)

    # remove searchers to free up resources
    db_id2searcher.clear()
    return returned_dataset

if __name__ == "__main__":
    """Prepare dataset for BIRD dev set only."""
    print("BIRD-dev")
    dataset_path = "./data/sft_data_collections/bird/dev/dev.json"
    table_path = "./data/sft_data_collections/bird/dev/dev_tables.json"
    if not os.path.exists(dataset_path) or not os.path.exists(table_path):
        print(
            f"Dataset or table schema not found at {dataset_path} / {table_path}, skipping preprocessing."
        )
    else:
        bird_dev = spider_style_dataset(
            dataset_path=dataset_path,
            db_path="./data/sft_data_collections/bird/dev/dev_databases",
            db_content_index_path="./data/sft_data_collections/bird/dev/db_contents_index",
            source="bird-dev",
            table_json_path=table_path,
            use_evidence=False,
            mode="dev",
        )
        with open("./data/sft_bird_dev_text2sql.json", "w") as f:
            f.write(json.dumps(bird_dev, indent=2, ensure_ascii=False))
