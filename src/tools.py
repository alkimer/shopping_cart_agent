# src/tools.py
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field


from functools import lru_cache
import os
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_current_user_id: Optional[int] = None


def set_user_id(uid: int):
    global _current_user_id
    _current_user_id = uid


def get_user_id() -> Optional[int]:
    return _current_user_id


# Global CSVs — loaded once, outside tool
PRODUCTS_CSV = "./dataset/products.csv"
DEPARTMENTS_CSV = "./dataset/departments.csv"

_products_df = pd.read_csv(PRODUCTS_CSV)
_product_lookup = dict(zip(_products_df["product_id"], _products_df["product_name"]))

# Load global CSVs once
products = pd.read_csv("./dataset/products.csv")
departments = pd.read_csv("./dataset/departments.csv")
aisles = pd.read_csv("./dataset/aisles.csv")
prior = pd.read_csv("./dataset/order_products__prior.csv")
orders = pd.read_csv("./dataset/orders.csv")


# Build enumerated options
DEPARTMENT_NAMES = sorted(departments["department"].dropna().unique().tolist())

VALID_USER_IDS = sorted(orders["user_id"].dropna().unique().tolist())

# Pick the first user for simplicity and safety
DEFAULT_USER_ID = VALID_USER_IDS[0]

# TODO
@tool
def structured_search_tool(
    product_name: Optional[str] = None,
    department: Optional[Literal[tuple(DEPARTMENT_NAMES)]] = None,
    aisle: Optional[str] = None,
    reordered: Optional[bool] = None,
    min_orders: Optional[int] = None,
    order_by: Optional[Literal["count", "add_to_cart_order"]] = None,
    ascending: Optional[bool] = False,
    top_k: Optional[int] = None,
    group_by: Optional[Literal["department", "aisle"]] = None,
    history_only: Optional[bool] = False,
) -> list:
    """
    A LangChain-compatible tool for structured product discovery across a grocery catalog and user purchase history.

    This function is decorated with `@tool` to expose it to an LLM agent via LangGraph. It supports SQL-like filtering
    over a product database, optionally constrained to the current user's order history. It returns either individual products
    or group-wise summaries based on the provided arguments.

    ---
    Tool Behavior Overview:
    - Operates on two global pandas DataFrames:
        • `products`: full catalog (from `products.csv`)
        • `prior` + `orders`: user order history (from `order_products__prior.csv`, `orders.csv`)
    - Uses additional joins with `departments.csv` and `aisles.csv` to enrich the product metadata.
    - If `history_only=True`, it will:
        • Look up the current user ID from `get_user_id()`
        • Merge product purchases for that user
        • Calculate statistics like reorder count, order frequency, and cart placement
    - Applies filters conditionally based on which arguments are set.
    - Optionally groups results by department or aisle.

    ---
    Parameters:
    - `product_name` (str, optional): Case-insensitive substring match on product names.
      Example: "almond" → matches "Almond Milk", "Almond Butter".

    - `department` (str, optional): Exact match against department names (from `DEPARTMENT_NAMES`).
      Example: "beverages", "pantry", "produce".

    - `aisle` (str, optional): Lowercased match on aisle name. Example: "organic snacks", "soy milk".

    - `reordered` (bool, optional): Only meaningful if `history_only=True`.
      - `True` → return only products reordered at least once
      - `False` → return products bought once, never reordered

    - `min_orders` (int, optional): Only meaningful if `history_only=True`.
      Filters for items purchased this many times or more.

    - `order_by` (str, optional): Only meaningful if `history_only=True`.
      - `"count"` → total times product was ordered
      - `"add_to_cart_order"` → average position in cart

    - `ascending` (bool, optional): Whether to sort `order_by` field ascending (default is `False` = descending).

    - `top_k` (int, optional): After filtering and sorting, returns only the top K products.

    - `group_by` (str, optional): If set to `"department"` or `"aisle"`, aggregates and returns counts instead of product rows.

    - `history_only` (bool, optional):
      - If `True`, only includes items the current user has purchased.
      - If `False` (default), searches the full catalog.

    ---
    Dependencies:
    - Requires global variables: `products`, `departments`, `aisles`, `prior`, `orders`
    - Requires user ID to be set via `set_user_id(user_id)` if `history_only=True`
    - Reads from CSVs under `./dataset/`

    ---
    Examples:
    ➤ Example 1: Find catalog items in pantry containing "peanut"
    ```json
    {
        "product_name": "peanut",
        "department": "pantry"
    }
    ```

    ➤ Example 2: Show reordered pantry products in my history
    ```json
    {
        "department": "pantry",
        "reordered": true,
        "history_only": true
    }
    ```

    ➤ Example 3: Top 5 most frequent purchases by user
    ```json
    {
        "order_by": "count",
        "top_k": 5,
        "history_only": true
    }
    ```

    ➤ Example 4: Count of catalog items by department
    ```json
    {
        "group_by": "department"
    }
    ```

    ---
    Returns:
    - If `group_by` is used:
      A list of dicts like:
      ```json
      [{"department": "pantry", "num_products": 132}, {"department": "beverages", "num_products": 89}]
      ```

    - Otherwise:
      A list of dicts, each with:
      ```json
      {
        "product_id": 24852,
        "product_name": "Organic Bananas",
        "aisle": "fresh fruits",
        "department": "produce",
        ... (optionally "count", "reordered", etc. if history_only=True)
      }
      ```

    - If no matches found, returns an empty list.
    - If required fields (e.g. user ID) are missing, returns a list with an error dict.

    ---
    LLM Usage Note:
    This tool is ideal for filtered browsing, purchase history analysis, or category breakdowns.
    """
    """
        Estructurada: filtra el catálogo y/o historial del usuario, con agregaciones opcionales.
        Devuelve una lista de dicts (o error en un dict) según los parámetros.
        """
    try:
        # -------- 1) Copias y normalización de tipos para merges --------
        p = products.copy()
        a = aisles.copy()
        d = departments.copy()

        for df, col in [(p, "aisle_id"), (p, "department_id"),
                        (a, "aisle_id"), (d, "department_id")]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        p = p.dropna(subset=["aisle_id", "department_id"]).copy()
        p["aisle_id"] = p["aisle_id"].astype("int64")
        p["department_id"] = p["department_id"].astype("int64")
        a["aisle_id"] = a["aisle_id"].astype("int64")
        d["department_id"] = d["department_id"].astype("int64")

        # -------- 2) Catálogo enriquecido --------
        catalog = (
            p.merge(a, on="aisle_id", how="left", validate="many_to_one")
            .merge(d, on="department_id", how="left", validate="many_to_one")
        )
        catalog["product_name"] = catalog["product_name"].astype(str)
        catalog["aisle"] = catalog["aisle"].astype(str).str.lower()
        catalog["department"] = catalog["department"].astype(str).str.lower()

        # -------- 3) Modo historial del usuario --------
        if history_only:
            uid = get_user_id() if get_user_id() is not None else DEFAULT_USER_ID

            ords = orders.copy()
            ords = ords[ords["user_id"] == uid]
            ords["eval_set"] = ords["eval_set"].astype(str).str.strip().str.lower()
            ords = ords[ords["eval_set"] == "prior"]
            if ords.empty:
                return [{"error": f"No order history found for user_id={uid}."}]

            op = prior.copy()
            op["order_id"] = pd.to_numeric(op["order_id"], errors="coerce")
            op["product_id"] = pd.to_numeric(op["product_id"], errors="coerce")
            op = op.dropna(subset=["order_id", "product_id"]).copy()
            op["order_id"] = op["order_id"].astype("int64")
            op["product_id"] = op["product_id"].astype("int64")

            user_lines = op.merge(
                ords[["order_id"]], on="order_id", how="inner", validate="many_to_one"
            )
            if user_lines.empty:
                return [{"error": f"User {uid} has no prior order lines."}]

            agg = user_lines.groupby("product_id", as_index=False).agg(
                count=("product_id", "size"),
                reordered_count=("reordered", "sum"),
                add_to_cart_order=("add_to_cart_order", "mean"),
            )

            df = agg.merge(
                catalog[["product_id", "product_name", "aisle", "department"]],
                on="product_id",
                how="left",
                validate="many_to_one",
            )

            # -------- 4) Filtros --------
            if product_name:
                q = str(product_name).lower()
                df = df[df["product_name"].str.lower().str.contains(q, na=False)]

            if department:
                df = df[df["department"].str.lower() == str(department).lower()]

            if aisle:
                df = df[df["aisle"].str.lower() == str(aisle).lower()]

            if reordered is True:
                df = df[df["reordered_count"] > 0]
            elif reordered is False:
                df = df[(df["count"] >= 1) & (df["reordered_count"] == 0)]

            if isinstance(min_orders, int) and min_orders > 0:
                df = df[df["count"] >= min_orders]

            # -------- 5) Orden, top_k, group_by --------
            if order_by in {"count", "add_to_cart_order"}:
                df = df.sort_values(by=order_by, ascending=bool(ascending), kind="mergesort")

            if isinstance(top_k, int) and top_k > 0:
                df = df.head(top_k)

            if group_by in {"department", "aisle"}:
                grp = (
                    df.groupby(group_by, as_index=False)
                    .agg(num_products=("product_id", "nunique"))
                    .sort_values("num_products", ascending=False)
                )
                return grp.to_dict(orient="records")

            # -------- 6) Salida (mantener 'count' para tests) --------
            cols = [
                "product_id",
                "product_name",
                "aisle",
                "department",
                "count",
                "reordered_count",
                "add_to_cart_order",
            ]
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df[cols].to_dict(orient="records")

        # -------- 7) Catálogo completo (sin historial) --------
        df = catalog.copy()

        if product_name:
            q = str(product_name).lower()
            df = df[df["product_name"].str.lower().str.contains(q, na=False)]

        if department:
            df = df[df["department"].str.lower() == str(department).lower()]

        if aisle:
            df = df[df["aisle"].str.lower() == str(aisle).lower()]

        if group_by in {"department", "aisle"}:
            grp = (
                df.groupby(group_by, as_index=False)
                .agg(num_products=("product_id", "nunique"))
                .sort_values("num_products", ascending=False)
            )
            if isinstance(top_k, int) and top_k > 0:
                grp = grp.head(top_k)
            return grp.to_dict(orient="records")

        if isinstance(top_k, int) and top_k > 0:
            df = df.head(top_k)

        return df[["product_id", "product_name", "aisle", "department"]].to_dict(orient="records")

    except Exception as e:
        return [{"error": f"structured_search_tool failed: {repr(e)}"}]


# TODO
class RouteToCustomerSupport(BaseModel):
    """
    Pydantic schema for the assistant tool that triggers routing to customer support.

    This tool is used by the assistant to signal that the user has a problem beyond
    the scope of sales, such as refund requests or broken products.

    ---
    Fields:
    - reason (str): A short, human-readable message stating why support is needed.
      This must match the user's stated concern.

    ---
    Usage Requirements:
    - The assistant must populate this tool with the user's reason verbatim.
    - It must be called in tool_calls from the LLM when escalation is needed.
    - This tool is detected by `after_sales_tool(...)` to drive state transitions.

    ---
    Example:
    ```json
    {
        "reason": "My laptop arrived broken and I want a refund"
    }
    ```

    This schema must be registered as a tool in the assistant's tool list.
    """

    reason: str = Field(description="The reason why the customer needs support.")


class EscalateToHuman(BaseModel):
    severity: str = Field(
        description="The severity level of the issue (low, medium, high)."
    )
    summary: str = Field(description="A brief summary of the customer's issue.")


# ---- NEW: Search tool and handler ----
class Search(BaseModel):
    query: str = Field(description="User's natural language product search query.")


CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"
_embeddings = None
_vector_store = None


def get_vector_store():
    global _embeddings, _vector_store

    if _vector_store is not None:
        return _vector_store

    if not os.path.isdir(CHROMA_DIR):
        raise RuntimeError(
            f"No se encontró la base vectorial en '{CHROMA_DIR}'. "
            f"Ejecutá primero el script de construcción."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    _vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_embeddings,
        persist_directory=CHROMA_DIR,
    )
    return _vector_store

def make_query_prompt(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query.strip().replace(chr(10), ' ')}"


# TODO
def search_products(query: str, top_k: int = 5):
    """
    Perform a semantic vector search over the product catalog using HuggingFace embeddings and Chroma.

    This function enables retrieval-augmented generation (RAG) by embedding a user's query and searching
    for the most relevant product entries using vector similarity.

    ---
    Requirements:
    - You must call `make_query_prompt(query: str) -> str` to wrap the query text.
    - You must call `get_vector_store()` to obtain a Chroma instance.
    - You must perform `similarity_search(query_text: str, k: int)` on the Chroma vector store.
    - Each result is a `Document` with `metadata` and `page_content`.

    ---
    Arguments:
    - query (str): A user query like "I want healthy snacks" or "almond milk".
    - top_k (int): Number of similar results to return (default: 5).

    ---
    Returns:
    - A list of dicts, each with the following fields:
        - "product_id" (int)
        - "product_name" (str)
        - "aisle" (str)
        - "department" (str)
        - "text" (str): full `page_content` of the result

    ---
    Example return:
    ```python
    [
        {
            "product_id": 123,
            "product_name": "Organic Almond Milk",
            "aisle": "Dairy Alternatives",
            "department": "Beverages",
            "text": "Organic Almond Milk, found in the Dairy Alternatives aisle..."
        }
    ]
    ```
    """
    """
    Perform a semantic vector search over the product catalog using HuggingFace embeddings and Chroma.
    """
    # Validación básica
    if not isinstance(query, str) or not query.strip():
        return []

    # 1) Envolver la consulta
    query_text = make_query_prompt(query)

    # 2) Obtener el vector store
    store = get_vector_store()

    # 3) Ejecutar similarity_search
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 5
    docs = store.similarity_search(query_text, k=k)

    # 4) Transformar a la salida requerida
    results = []
    for d in docs:
        md = d.metadata or {}
        results.append({
            "product_id": int(md["product_id"]) if md.get("product_id") is not None else None,
            "product_name": md.get("product_name", "") or "",
            "aisle": md.get("aisle", "") or "",
            "department": md.get("department", "") or "",
            "text": d.page_content or "",
        })
    return results


# TODO
@tool
def search_tool(query: str) -> str:
    """
    Tool-decorated function that performs semantic product search using vector similarity,
    formats the results into a human-readable response, and is callable by a LangChain agent.

    This function is registered as a LangChain tool using the `@tool` decorator. It is intended
    for natural language queries from users looking for relevant products. Internally, it wraps
    `search_products(...)`, which uses a sentence embedding model and vector database.

    ---
    Tool Decorator:
    - This function is wrapped with `@tool` so that it can be invoked by an LLM agent during
      LangGraph execution when choosing from available tools.

    ---
    Arguments:
    - query (str): A free-form natural language string describing what the user is looking for.
      Examples:
        - "high protein vegan snacks"
        - "easy breakfast foods"
        - "organic almond butter"

    ---
    Internal Behavior:
    - Calls `search_products(query: str)` to perform semantic vector search.
    - The `search_products()` function:
        • Uses `make_query_prompt()` to convert the query into a format suitable for embedding.
        • Embeds the prompt using a HuggingFace sentence transformer.
        • Calls `get_vector_store()` to get a Chroma DB.
        • Returns metadata-rich matches including ID, name, aisle, department, and description.
    - The results (if any) are converted into a multi-line formatted string showing:
        - Product name and ID
        - Aisle and Department
        - Text description from vector DB

    ---
    Returns:
    - If products found:
        A formatted multiline string:
        ```
        - Organic Granola (ID: 18872)
          Aisle: Cereal
          Department: Breakfast
          Details: Organic Granola, found in the Cereal aisle...
        ```
    - If no products found:
        `"No products found matching your search."`

    ---
    Example Use (from an LLM):
    ```python
    search_tool("something high protein for breakfast")
    ```
    """
    # Ejecutar la búsqueda semántica
    hits = search_products(query)

    # Sin resultados
    if not hits:
        return "No products found matching your search."

    # Formateo legible
    lines = []
    for h in hits:
        pid = h.get("product_id")
        pname = h.get("product_name", "")
        aisle = h.get("aisle", "")
        dept = h.get("department", "")
        text = h.get("text", "")

        lines.append(
            f"- {pname} (ID: {pid})\n"
            f"  Aisle: {aisle}\n"
            f"  Department: {dept}\n"
            f"  Details: {text}"
        )

    return "\n".join(lines)

# ---- UPDATED: Cart tools with quantity support ----
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Simulated in-memory cart storage - now stores product_id: quantity pairs
_cart_storage: Dict[str, Dict[int, int]] = {}
_current_thread_id: Optional[str] = None


def set_thread_id(tid: str):
    global _current_thread_id
    _current_thread_id = tid


def get_cart() -> Union[List[str], Dict[int, int]]:
    if _current_thread_id is None:
        return ["Session error: no thread ID set."]
    return _cart_storage.setdefault(_current_thread_id, {})


@tool
def cart_tool(
    cart_operation: str, product_id: Optional[int] = None, quantity: int = 1
) -> str:
    """
    Modify the user's cart by adding, removing, updating quantity, or buying products.

    Args:
        cart_operation: The operation to perform (add, remove, update, buy)
        product_id: The ID of the product to add, remove, or update
        quantity: The quantity for add or update operations (default: 1)
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if cart_operation == "add":
        if product_id is None:
            return "No product ID provided to add."

        # If product already exists, increase quantity
        if product_id in cart:
            cart[product_id] += quantity
            return f"Added {quantity} more of product {product_id} to your cart. New quantity: {cart[product_id]}."
        else:
            cart[product_id] = quantity
            product_name = _product_lookup.get(product_id, "Unknown Product")
            return (
                f"Added {quantity} of {product_name} (ID: {product_id}) to your cart."
            )

    elif cart_operation == "update":
        if product_id is None:
            return "No product ID provided to update."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        cart[product_id] = quantity
        product_name = _product_lookup.get(product_id, "Unknown Product")
        return f"Updated quantity of {product_name} (ID: {product_id}) to {quantity}."

    elif cart_operation == "remove":
        if product_id is None:
            return "No product ID provided to remove."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        product_name = _product_lookup.get(product_id, "Unknown Product")

        # If quantity is specified and less than current quantity, reduce quantity
        if quantity > 1 and cart[product_id] > quantity:
            cart[product_id] -= quantity
            return f"Removed {quantity} of {product_name} (ID: {product_id}) from your cart. New quantity: {cart[product_id]}."
        else:
            # Otherwise remove product completely
            del cart[product_id]
            return f"Removed {product_name} (ID: {product_id}) from your cart."

    elif cart_operation == "buy":
        if not cart:
            return "Your cart is empty. Nothing to purchase."
        cart.clear()
        return "Thank you for your purchase! Your cart is now empty."

    return f"Unknown cart operation: {cart_operation}"


@tool
def view_cart() -> str:
    """
    Display the contents of the user's cart with quantities.
    This is a standard tool that returns a formatted string representation of the cart.
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if not cart:
        return "Your cart is currently empty."

    lines = ["Your cart contains:"]
    for pid, qty in cart.items():
        title = _product_lookup.get(pid, "Unknown Product")
        lines.append(f"- {title} (ID: {pid}) × {qty}")
    return "\n".join(lines)


# ---- Tool fallback handling ----


def handle_tool_error(state: Dict[str, Any]) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistake.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# TODO
def create_tool_node_with_fallback(tools: list) -> ToolNode:
    """
    Build a LangGraph ToolNode that can handle errors gracefully using a fallback strategy.

    This function should create a ToolNode that wraps a list of tools and attaches
    a fallback mechanism using LangChain's `with_fallbacks(...)` method.

    ---
    Requirements:
    - Return a `ToolNode` from `langgraph.prebuilt`.
    - Attach a fallback using `.with_fallbacks(...)` with your error handler.
    - Use `handle_tool_error(state)` as the fallback function.
    - Set `exception_key="error"` so LangGraph recognizes failure states.

    ---
    Arguments:
    - tools (list): A list of @tool-decorated functions (LangChain tools).

    ---
    Returns:
    - ToolNode: A LangGraph-compatible tool node with error fallback logic.
    """
    node = ToolNode(tools)
    # Attach a fallback runnable that will be called when an exception is stored under "error"
    return node.with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


__all__ = [
    "RouteToCustomerSupport",
    "EscalateToHuman",
    "Search",
    "search_products",
    "search_tool",
    "cart_tool",
    "view_cart",
    "handle_tool_error",
    "create_tool_node_with_fallback",
    "structured_search_tool",
]
