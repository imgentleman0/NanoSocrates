import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, unquote
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
# Predicates that we take to create the RDF triples. We take only the most informative
# predicates, that are also the most common ones.
DEFAULT_PREDICATES = [
    "http://dbpedia.org/ontology/director",
    "http://dbpedia.org/ontology/starring",
    "http://dbpedia.org/ontology/releaseDate",
    "http://dbpedia.org/ontology/genre",
    "http://dbpedia.org/ontology/producer",
    "http://dbpedia.org/ontology/writer",
    "http://dbpedia.org/ontology/musicComposer",
    "http://dbpedia.org/ontology/runtime",
]

# Special tokens to insert inside the samples (as requested in the exam track)
SPECIAL_TOKENS = [
    "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
    "<RDF2Text>", "<Text2RDF>", "<CONTINUERDF>", "<MASK>"
]

# We ask the server SPARQL results in JSON
HEADERS_SPARQL = {"Accept": "application/sparql-results+json"}
# JSON results, "User-Agent" is a good practice for public APIs
HEADERS_WIKI = {"Accept": "application/json", "User-Agent": "DBpedia Movies Dataset Creator/1.0"}

# We set the SPARQL endpoint of DBPedia
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
# REST endpoint of Wikipedia
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# This function creates a requests.Session(), configuring a retry policy and mounting it on both http:// and https:// schemas
def http_session(total_retries: int = 3, backoff: float = 0.5, timeout: int = 30) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff, # Exponential backoff between the retries
        status_forcelist=(429, 500, 502, 503, 504), # Makes another request if 429 (rate limit) or 5xx, due to server errors
        allowed_methods=frozenset(["GET", "POST"]), # We abilitate retry only on this two methods
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries) # Provides a general-case interface for Requests sessions to contact HTTP and HTTPS urls
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request_timeout = timeout
    return s


def sparql_select(session: requests.Session, query: str) -> Dict:
    params = {"query": query, "format": "application/sparql-results+json"} # query is the SPARQL string, asking explicitly the
    # JSON format of the results
    resp = session.get(DBPEDIA_SPARQL, params=params, headers=HEADERS_SPARQL, timeout=session.request_timeout)
    resp.raise_for_status()
    return resp.json()

# This function takes a text (in this case, the Wikipedia Abstract), cleans it
# and returns only the paragraph, without any citation (like, [12])
def clean_first_paragraph(text: str) -> str:
    text = text.strip() # Removes spaces, tabs and newlines at the start and at the end
    text = re.sub(r"\s*\[\d+\]\s*", " ", text)
    parts = re.split(r"\n\s*\n", text) # Splits the text in paragraphs, using one empty row as a
    # separator
    return parts[0].strip() # returns the first paragraph cleaned

# This function converts a DBpedia URI in compact form, said qname
def qname(uri: str) -> str:
    if uri.startswith("http://dbpedia.org/resource/"):
        return "dbr:" + uri.split("/")[-1] # Takes the last part of the split
    if uri.startswith("http://dbpedia.org/ontology/"):
        return "dbo:" + uri.split("/")[-1]
    if uri.startswith("http://dbpedia.org/property/"):
        return "dbp:" + uri.split("/")[-1]
    return uri

# Prepares a string to be used inside a literal between double quotes
def format_literal(val: str) -> str:
    escaped = val.replace('"', '\\"')
    return f"\"{escaped}\""

# It creates the directory p if it doesn't already exist
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# Writes a list of dictionaries in a JSON Lines file (.jsonl), one row for
# each record.
def write_jsonl(path: str, rows: List[Dict]):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Selects film in pseudo-random reproducible order, using a seed
def select_films(session: requests.Session, limit: int, offset: int = 0, seed: str = "42") -> List[str]:
    # dbo:Film takes only resources labeled as Films
    # dbo:abstract with the filter takes only films that have an abstract in English
    # The idea is to avoid multiple languages and multiple characters from different alphabets
    # MD5 calculates an exadecimal hash from the resulting string, used to order the films
    # The offset is used to go to other 'pages' to choose the films
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?film WHERE {{
      ?film a dbo:Film . 
      ?film dbo:abstract ?abs .
      FILTER (lang(?abs) = "en")
      BIND(MD5(CONCAT(STR(?film), "{seed}")) AS ?h)
    }}
    ORDER BY ?h
    LIMIT {limit}
    OFFSET {offset}
    """
    data = sparql_select(session, query)
    return [b["film"]["value"] for b in data["results"]["bindings"]]


# This function gets the Wikipedia page title associated to a DBpedia film
def get_wikipedia_title(session: requests.Session, movie_uri: str) -> Optional[str]:
    query = f"""
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?wikipediaPage WHERE {{
        <{movie_uri}> foaf:isPrimaryTopicOf ?wikipediaPage .
        FILTER(STRSTARTS(STR(?wikipediaPage), "https://en.wikipedia.org/"))
    }}
    LIMIT 1
    """
    # foaf:isPrimaryTopicOF links the DBpedia resource to the corresponding Wikipedia page
    # filter ensures that the Wikipedia page is in English
    try:
        data = sparql_select(session, query)
        bindings = data.get("results", {}).get("bindings", [])
        if bindings: # If there is at least one binding, gets the last part after /
            # and passes it to unquote(...) to URL-decode percent-encoded characters (e.g. %C3%A9 → é).
            wiki_url = bindings[0]["wikipediaPage"]["value"]
            title = wiki_url.split("/")[-1]
            return unquote(title)
    except Exception:
        pass
    if movie_uri.startswith("http://dbpedia.org/resource/"):
        return unquote(movie_uri.rsplit("/", 1)[-1])
    return None

# This function retreives the Wikipedia abstract associated to a movies' title
def fetch_abstract_wikipedia(session: requests.Session, film_uri: str) -> Optional[str]:
    title = get_wikipedia_title(session, film_uri) # Finds the title for the English Wikipedia page associated with film_uri
    if not title:
        return None
    url = WIKI_SUMMARY + quote(title) # URL-encodes the page title and appends it to Wikipedia's summary endpoint
    resp = session.get(url, headers=HEADERS_WIKI, timeout=session.request_timeout) # It performs a GET using the
    # session and the 'User-Agent' we declared before
    if not resp.ok:
        return None
    try:
        js = resp.json()
    except Exception:
        return None
    text = (js.get("extract") or js.get("description") or "").strip()
    if not text:
        return None
    parts = re.split(r"\n\s*\n", text) # Splits on blank lines and returns the first paragraph
    return parts[0].strip()


def fetch_filtered_triples(session: requests.Session, film_uri: str, predicates: List[str]) -> List[Tuple[str, str, str]]:
    values_pred = " ".join(f"<{p}>" for p in predicates)
    query = f"""
    SELECT ?p ?o WHERE {{
      VALUES ?p {{ {values_pred} }}
      <{film_uri}> ?p ?o .
    }}
    """
    # <{film_uri}> ?p ?o keeps all the prdicate/object couples allowed
    data = sparql_select(session, query)
    triples: List[Tuple[str, str, str]] = []
    seen = set()
    for b in data["results"]["bindings"]:
        p = b["p"]["value"]
        ob = b["o"]
        if ob["type"] == "uri":
            o = ob["value"]
        elif ob["type"] == "literal":
            lang = ob.get("xml:lang")
            if lang and lang != "en": # We keep only english language movies
                continue
            o = ob["value"]
            if len(o) > 300: # We truncate if the text exceed 300 characters
                o = o[:297] + "..."
        else:
            continue
        key = (film_uri, p, o) # We avoid exact duplicates of (s,p,o)
        if key in seen:
            continue
        seen.add(key)
        triples.append(key)

    def sort_key(t): # We order by predicate, converting the DBpedia URI in compact forms,
        _s, p, o = t
        pq = qname(p)
        oq = qname(o) if isinstance(o, str) and o.startswith("http") else o
        return (pq, oq)
    triples.sort(key=sort_key) # Sort by predicate and after by object
    return triples

# We serialize the triples for the tasks. The serialization follows the structure described in the exam track
def serialize_triples(triples: List[Tuple[str, str, str]]) -> str:
    parts = []
    for (s, p, o) in triples:
        s_q = qname(s)
        p_q = qname(p)
        if isinstance(o, str) and o.startswith("http"):
            o_q = qname(o)
        else:
            o_q = format_literal(o) if isinstance(o, str) else o
        parts.append(f"<SOT><SUBJ>{s_q}<PRED>{p_q}<OBJ>{o_q}<EOT>")
    return "".join(parts)


# We create the examples for the RDF Completion 1 task.
# We mask randomly one of the possible roles in the triple. The masked part of the triple
 # will be set as the target.
def make_completion1(triples: List[Tuple[str, str, str]], rng: random.Random) -> Optional[Dict]:
    if not triples:
        return None

    s, p, o = rng.choice(triples) # Here we choose the role to mask
    role = rng.choice(["SUBJ", "PRED", "OBJ"])

    o_maskable = qname(o) if isinstance(o, str) and o.startswith("http") else format_literal(o)

    # We distinguish the cases for the possible roles to mask
    if role == "SUBJ":
        masked_triple = f"<SOT><MASK><PRED>{qname(p)}<OBJ>{o_maskable}<EOT>"
        tgt = f"<SUBJ>{qname(s)}"
    elif role == "PRED":
        masked_triple = f"<SOT><SUBJ>{qname(s)}<MASK><OBJ>{o_maskable}<EOT>"
        tgt = f"<PRED>{qname(p)}"
    else:  # OBJ
        masked_triple = f"<SOT><SUBJ>{qname(s)}<PRED>{qname(p)}<MASK><EOT>"
        tgt = f"<OBJ>{qname(o)}"

    return {"input": masked_triple, "target": tgt}

# We create the samples for the RDF Completion 2 task. We take all the triples associated to a film,
# and we take a random k. The triples from the first to the k-th will be selected as input, while the triples
# from the k-th to the last will be selected as target.
def make_completion2(triples: List[Tuple[str, str, str]], rng: random.Random) -> Optional[Dict]:
    n = len(triples)
    if n < 2:
        return None
    k = rng.randint(1, n - 1)
    ctx = serialize_triples(triples[:k])
    tgt = serialize_triples(triples[k:])
    return {"input": f"{ctx}<CONTINUERDF>", "target": tgt, "k": k}

@dataclass
class Example: # Class Example to create samples in the build_dataset function
    entity: str
    text: str
    triples: List[Tuple[str, str, str]]
    rdf_serialized: str
    tasks: Dict[str, Dict]

# This function is used to build the single samples for the four tasks, returning
# a list of rows {"input", "target"}
def examples_to_unified_rows(exs: List["Example"]) -> List[Dict]:
    rows: List[Dict] = []
    for ex in exs:
        # RDF2Text
        rows.append({
            "input":  ex.tasks["rdf2text"]["input"],
            "target": ex.tasks["rdf2text"]["target"],
        })
        # Text2RDF
        rows.append({
            "input":  ex.tasks["text2rdf"]["input"],
            "target": ex.tasks["text2rdf"]["target"],
        })
        # Completion1
        if ex.tasks.get("completion1"):
            rows.append({
                "input":  ex.tasks["completion1"]["input"],
                "target": ex.tasks["completion1"]["target"],
            })
        # Completion2
        if ex.tasks.get("completion2"):
            rows.append({
                "input":  ex.tasks["completion2"]["input"],
                "target": ex.tasks["completion2"]["target"],
            })
    return rows

# This is used to save the special tokens and create the dataset
def save_unified(outdir: str, examples: List["Example"]):
    ensure_dir(outdir)
    with open(os.path.join(outdir, "special_tokens.txt"), "w", encoding="utf-8") as f:
        for tok in SPECIAL_TOKENS:
            f.write(tok + "\n")

    rows = examples_to_unified_rows(examples)
    out_path = Path(outdir) / "dataset.jsonl"
    write_jsonl(out_path, rows)

# This function expands the predicates, building the full URL considering the prefix
def expand_predicates(preds: List[str]) -> List[str]:
    out = []
    for p in preds:
        if p.startswith("dbo:"):
            out.append("http://dbpedia.org/ontology/" + p.split(":", 1)[1])
        elif p.startswith("dbp:"):
            out.append("http://dbpedia.org/property/" + p.split(":", 1)[1])
        elif p.startswith("dbr:"):
            out.append("http://dbpedia.org/resource/" + p.split(":", 1)[1])
        elif p.startswith("http"):
            out.append(p)
        else:
            out.append("http://dbpedia.org/ontology/" + p)
    return out

# This is the core of the script, indeed this function builds the dataset in JSONL format
# used to train and validate the model
def build_dataset(limit, # How many films to process
                  page_size, # How many films per "page" via SPARQL
                  predicates, # Predicates to extract
                  sleep_sec, # Pause between films (throttling against rate limit)
                  seed, # Used for the pseudo-random choice of the films, for MASK tokens and for k in RC2
                  max_triples, # Max number of triples per film
                  timeout,  # Last three parameters for HTTP robustness
                  retries,
                  backoff) -> List[Example]:
    rng = random.Random(seed)
    session = http_session(total_retries=retries, backoff=backoff, timeout=timeout)
    examples: List[Example] = []

    for off in range(0, limit, page_size): # Block-by-block (OFFSET / LIMIT) iterates through DBpedia movies
        batch = min(page_size, limit - off)
        uris = select_films(session, limit=batch, offset=off, seed=str(seed)) # As said before, select_films sort the movies
        # in a pseudo-random way
        if not uris:
            break

        for film in tqdm(uris, desc=f"Processing {off+1}-{off+batch}"):
            try:
                abs_text = fetch_abstract_wikipedia(session, film)
                if not abs_text: # If there isn't an abstract available, skip the film
                    continue
                text = clean_first_paragraph(abs_text)

                triples = fetch_filtered_triples(session, film, predicates)
                if not triples: # If there aren't triples available, skip the film
                    continue
                if max_triples: # If the triples exceed the maximum number, we cut it
                    triples = triples[:max_triples]


                # For each film, we create an example for each task, so
                # basically, from a single film we create 4 samples
                rdf_ser = serialize_triples(triples)
                t2r_input = f"{text}<Text2RDF>"
                r2t_input = f"{rdf_ser}<RDF2Text>"

                comp1 = make_completion1(triples, rng)
                comp2 = make_completion2(triples, rng)

                examples.append(Example(
                    entity=qname(film),
                    text=text,
                    triples=triples,
                    rdf_serialized=rdf_ser,
                    tasks={
                        "text2rdf": {"input": t2r_input, "target": rdf_ser},
                        "rdf2text": {"input": r2t_input, "target": text},
                        "completion1": comp1,
                        "completion2": comp2,
                    }
                ))

                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            # Warning on errors
            except requests.RequestException as e:
                print(f"[WARN] {film} -> {e}")
                continue

    return examples

def main():
    ap = argparse.ArgumentParser(description="Build NanoSocrates dataset from DBpedia and Wikipedia into a JSONL file.")
    ap.add_argument("--limit", type=int, default=350, help="Total number of films to fetch.")
    ap.add_argument("--page-size", type=int, default=50, help="Pagination for SPARQL queries.")
    ap.add_argument("--outdir", type=str, default="data", help="Output directory.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between film requests.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for masking and selection order.")
    ap.add_argument("--max-triples", type=int, default=None, help="Optional cap per entity.")
    ap.add_argument("--predicates", nargs="*", default=[], help="Override predicates, e.g., dbo:director dbo:starring dbo:releaseDate dbo:genre")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    ap.add_argument("--retries", type=int, default=3, help="HTTP total retries for 429/5xx.")
    ap.add_argument("--backoff", type=float, default=0.5, help="HTTP backoff factor.")
    args = ap.parse_args()

    predicates = expand_predicates(args.predicates if args.predicates else DEFAULT_PREDICATES)

    examples = build_dataset(
        limit=args.limit,
        page_size=args.page_size,
        predicates=predicates,
        sleep_sec=args.sleep,
        seed=args.seed,
        max_triples=args.max_triples,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )

    args.outdir = str(ROOT / args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    save_unified(args.outdir, examples)

    print(f"Wrote unified dataset to {Path(args.outdir) / 'dataset.jsonl'}.")
if __name__ == "__main__":
    main()
